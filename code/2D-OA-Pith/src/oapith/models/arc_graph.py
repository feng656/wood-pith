from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from oapith.types import ArcObservation


@dataclass
class ArcGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    same_ring_target: torch.Tensor


def _curvature(points: torch.Tensor) -> torch.Tensor:
    first = points[1:-1] - points[:-2]
    second = points[2:] - points[1:-1]
    first = torch.nn.functional.normalize(first, dim=-1, eps=1e-8)
    second = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    angle = torch.acos((first * second).sum(-1).clamp(-1, 1))
    spacing = 0.5 * (
        torch.linalg.vector_norm(points[1:-1] - points[:-2], dim=-1)
        + torch.linalg.vector_norm(points[2:] - points[1:-1], dim=-1)
    )
    return angle / spacing.clamp_min(1e-6)


def build_arc_graph(arcs: list[ArcObservation], maximum_endpoint_distance: float = 0.35) -> ArcGraph:
    """Build center-independent node/edge evidence for reliability and connection."""
    if not arcs:
        raise ValueError("arcs are empty")
    nodes = []
    for arc in arcs:
        length = torch.linalg.vector_norm(arc.points[1:] - arc.points[:-1], dim=-1).sum()
        curvature = _curvature(arc.points)
        tangent_scatter = arc.tangents.T @ arc.tangents / arc.tangents.shape[0]
        tangent_eigenvalues = torch.linalg.eigvalsh(tangent_scatter)
        sigma = arc.sigma.mean() if arc.sigma.ndim else arc.sigma
        nodes.append(
            torch.stack(
                [
                    torch.log(length.clamp_min(1e-6)),
                    torch.log(length.new_tensor(arc.points.shape[0])),
                    curvature.mean(),
                    curvature.std(correction=0),
                    curvature.max(),
                    sigma,
                    length.new_tensor(arc.prior_reliability),
                    tangent_eigenvalues[0],
                    length.new_tensor(float(arc.metadata.get("defect_overlap", 0.0))),
                    length.new_tensor(float(arc.metadata.get("endpoint_quality", 1.0))),
                ]
            )
        )
    edges, features, targets = [], [], []
    for first in range(len(arcs)):
        for second in range(first + 1, len(arcs)):
            a, b = arcs[first], arcs[second]
            endpoints_a = torch.stack([a.points[0], a.points[-1]])
            endpoints_b = torch.stack([b.points[0], b.points[-1]])
            distances = torch.cdist(endpoints_a, endpoints_b)
            flat_index = int(distances.argmin())
            endpoint_distance = distances.reshape(-1)[flat_index]
            endpoint_a = flat_index // 2
            endpoint_b = flat_index % 2
            tangent_alignment = (
                a.tangents[[0, -1][endpoint_a]] * b.tangents[[0, -1][endpoint_b]]
            ).sum().abs()
            if a.embedding is not None and b.embedding is not None:
                embedding_similarity = torch.nn.functional.cosine_similarity(
                    a.embedding, b.embedding, dim=0
                )
            else:
                embedding_similarity = endpoint_distance.new_zeros(())
            if endpoint_distance > maximum_endpoint_distance and embedding_similarity < 0.5:
                continue
            sigma_a = a.sigma.mean() if a.sigma.ndim else a.sigma
            sigma_b = b.sigma.mean() if b.sigma.ndim else b.sigma
            length_a = torch.linalg.vector_norm(a.points[1:] - a.points[:-1], dim=-1).sum()
            length_b = torch.linalg.vector_norm(b.points[1:] - b.points[:-1], dim=-1).sum()
            edge_feature = torch.stack(
                [
                    endpoint_distance,
                    tangent_alignment,
                    embedding_similarity,
                    torch.log((sigma_a + sigma_b).clamp_min(1e-6)),
                    endpoint_distance.new_tensor(abs(a.prior_reliability - b.prior_reliability)),
                    torch.abs(torch.log(length_a.clamp_min(1e-6)) - torch.log(length_b.clamp_min(1e-6))),
                ]
            )
            for source, destination in ((first, second), (second, first)):
                edges.append([source, destination])
                features.append(edge_feature)
                targets.append(float(a.ring_id == b.ring_id))
    device, dtype = arcs[0].points.device, arcs[0].points.dtype
    if not edges:
        edges = [[0, 0]]
        features = [torch.zeros(6, device=device, dtype=dtype)]
        targets = [0.0]
    return ArcGraph(
        node_features=torch.stack(nodes),
        edge_index=torch.tensor(edges, device=device, dtype=torch.long).T,
        edge_features=torch.stack(features),
        same_ring_target=torch.tensor(targets, device=device, dtype=dtype),
    )


class ArcGraphNet(nn.Module):
    """Message passing without any candidate-pith feature, avoiding self-confirming pruning."""

    def __init__(
        self,
        node_features: int = 10,
        edge_features: int = 6,
        hidden: int = 96,
        layers: int = 3,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(node_features, hidden), nn.SiLU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_features, hidden), nn.SiLU())
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 3, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(layers)
            ]
        )
        self.update_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU()) for _ in range(layers)]
        )
        self.node_reliability = nn.Linear(hidden, 1)
        self.same_ring = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, graph: ArcGraph) -> dict[str, torch.Tensor]:
        node = self.node_encoder(graph.node_features)
        edge = self.edge_encoder(graph.edge_features)
        source, destination = graph.edge_index
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            message = message_layer(torch.cat([node[source], node[destination], edge], dim=-1))
            aggregate = torch.zeros_like(node)
            aggregate.index_add_(0, destination, message)
            degree = torch.bincount(destination, minlength=node.shape[0]).clamp_min(1).to(node.dtype)
            aggregate = aggregate / degree[:, None]
            node = node + update_layer(torch.cat([node, aggregate], dim=-1))
        edge_logits = self.same_ring(
            torch.cat([node[source], node[destination], edge], dim=-1)
        ).squeeze(-1)
        return {
            "node_reliability_logits": self.node_reliability(node).squeeze(-1),
            "same_ring_logits": edge_logits,
        }


def apply_graph_priors(
    arcs: list[ArcObservation], graph: ArcGraph, output: dict[str, torch.Tensor], threshold: float = 0.5
) -> list[ArcObservation]:
    reliability = torch.sigmoid(output["node_reliability_logits"]).detach()
    for arc, value in zip(arcs, reliability):
        arc.prior_reliability = float(value.clamp(0.01, 0.995))
    parent = list(range(len(arcs)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for edge_number, probability in enumerate(torch.sigmoid(output["same_ring_logits"]).detach()):
        if float(probability) < threshold:
            continue
        source, destination = (int(value) for value in graph.edge_index[:, edge_number])
        a, b = find(source), find(destination)
        if a != b:
            parent[b] = a
    mapping: dict[int, int] = {}
    for index, arc in enumerate(arcs):
        root = find(index)
        arc.ring_id = mapping.setdefault(root, len(mapping))
    return arcs
