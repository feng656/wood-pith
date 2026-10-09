from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from oapith.geometry.curves import cumulative_arclength, resample_polyline
from oapith.types import ArcObservation, CoordinateFrame


def zhang_suen_thinning(binary: np.ndarray, max_iterations: int = 256) -> np.ndarray:
    """Topology-preserving thinning for a 2-D binary ridge mask."""
    image = np.pad(binary.astype(np.uint8), 1)
    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]
            center = image[1:-1, 1:-1]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum(neighbors)
            transitions = sum(
                ((neighbors[index] == 0) & (neighbors[(index + 1) % 8] == 1)).astype(np.uint8)
                for index in range(8)
            )
            remove = (center == 1) & (count >= 2) & (count <= 6) & (transitions == 1)
            if step == 0:
                remove &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                remove &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if remove.any():
                center[remove] = 0
                changed = True
        if not changed:
            break
    return image[1:-1, 1:-1].astype(bool)


_NEIGHBORS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _trace_skeleton(skeleton: np.ndarray, minimum_pixels: int) -> list[np.ndarray]:
    pixels = {tuple(value) for value in np.argwhere(skeleton)}
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pixel in pixels:
        adjacency[pixel] = [
            (pixel[0] + dy, pixel[1] + dx)
            for dy, dx in _NEIGHBORS
            if (pixel[0] + dy, pixel[1] + dx) in pixels
        ]
    nodes = [pixel for pixel, neighbors in adjacency.items() if len(neighbors) != 2]
    visited_edges: set[frozenset[tuple[int, int]]] = set()
    paths: list[np.ndarray] = []

    def follow(start: tuple[int, int], neighbor: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start, neighbor]
        visited_edges.add(frozenset((start, neighbor)))
        previous, current = start, neighbor
        while len(adjacency[current]) == 2:
            candidates = [value for value in adjacency[current] if value != previous]
            if not candidates:
                break
            nxt = candidates[0]
            edge = frozenset((current, nxt))
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            path.append(nxt)
            previous, current = current, nxt
        return path

    for node in nodes:
        for neighbor in adjacency[node]:
            if frozenset((node, neighbor)) not in visited_edges:
                path = follow(node, neighbor)
                if len(path) >= minimum_pixels:
                    paths.append(np.asarray(path, dtype=np.float64))
    # Closed loops have degree two everywhere and therefore no explicit node.
    for pixel, neighbors in adjacency.items():
        for neighbor in neighbors:
            if frozenset((pixel, neighbor)) not in visited_edges:
                path = follow(pixel, neighbor)
                if len(path) >= minimum_pixels:
                    paths.append(np.asarray(path, dtype=np.float64))
    return paths


def _sample(array: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(array, [row, col], order=1, mode="nearest")


def _subpixel_ridge(
    path_rc: np.ndarray, probability: np.ndarray, orientation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    row, col = path_rc[:, 0], path_rc[:, 1]
    cos2 = _sample(orientation[0], row, col)
    sin2 = _sample(orientation[1], row, col)
    theta = 0.5 * np.arctan2(sin2, cos2)
    tangent = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # x,y
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    center = _sample(probability, row, col)
    minus = _sample(probability, row - normal[:, 1], col - normal[:, 0])
    plus = _sample(probability, row + normal[:, 1], col + normal[:, 0])
    denominator = minus - 2.0 * center + plus
    offset = np.divide(
        0.5 * (minus - plus),
        denominator,
        out=np.zeros_like(center),
        where=np.abs(denominator) > 1e-6,
    )
    offset = np.clip(offset, -0.5, 0.5)
    xy = np.stack([col, row], axis=-1) + offset[:, None] * normal
    return xy, tangent


def _cluster_embeddings(means: list[np.ndarray], threshold: float) -> list[int]:
    parent = list(range(len(means)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[b] = a

    for first in range(len(means)):
        for second in range(first + 1, len(means)):
            denominator = max(np.linalg.norm(means[first]) * np.linalg.norm(means[second]), 1e-8)
            similarity = float(np.dot(means[first], means[second]) / denominator)
            if similarity >= threshold:
                union(first, second)
    mapping: dict[int, int] = {}
    return [mapping.setdefault(find(index), len(mapping)) for index in range(len(means))]


def _resample_scalar_field(
    source_points: torch.Tensor, values: torch.Tensor, target_points: torch.Tensor
) -> torch.Tensor:
    source = cumulative_arclength(source_points)
    target = cumulative_arclength(target_points)
    source = source / source[-1].clamp_min(1e-8)
    target = target / target[-1].clamp_min(1e-8)
    right = torch.searchsorted(source, target, right=True).clamp(1, source.shape[0] - 1)
    left = right - 1
    alpha = (target - source[left]) / (source[right] - source[left]).clamp_min(1e-8)
    return values[left] * (1 - alpha) + values[right] * alpha


def extract_arcs(
    ring_probability: torch.Tensor,
    orientation: torch.Tensor,
    log_variance: torch.Tensor,
    embedding: torch.Tensor,
    *,
    threshold: float = 0.35,
    minimum_pixels: int = 12,
    sample_spacing: float = 0.015,
    embedding_similarity: float = 0.88,
    distance_truncation: float = 0.04,
    frame: CoordinateFrame | None = None,
) -> list[ArcObservation]:
    """Probability ridge -> sub-pixel polylines -> ring-instance arc observations."""
    probability = ring_probability.detach().float().cpu().squeeze().numpy()
    orientation_np = orientation.detach().float().cpu().numpy()
    log_variance_np = log_variance.detach().float().cpu().squeeze().numpy()
    embedding_np = embedding.detach().float().cpu().numpy()
    if probability.ndim != 2 or orientation_np.shape[0] != 2:
        raise ValueError("invalid dense prediction shapes")
    if distance_truncation <= 0:
        raise ValueError("distance_truncation must be positive")
    skeleton = zhang_suen_thinning(probability >= threshold)
    paths = _trace_skeleton(skeleton, minimum_pixels)
    size_y, size_x = probability.shape
    frame = frame or CoordinateFrame(size_x, size_y)
    if frame.width != size_x or frame.height != size_y:
        raise ValueError("coordinate frame does not match dense prediction")
    candidates = []
    embedding_means = []
    for path in paths:
        xy_px, _ = _subpixel_ridge(path, probability, orientation_np)
        source_points = frame.pixel_to_normalized(torch.tensor(xy_px, dtype=torch.float64))
        points = resample_polyline(source_points, sample_spacing)
        # A short native-resolution skeleton may contain many adjacent pixels but
        # collapse to fewer than three samples at the normalized spacing.  It is
        # not a valid geometric arc and must not reach ArcObservation.validate().
        if points.shape[0] < 3:
            continue
        row = np.clip(np.rint(path[:, 0]).astype(int), 0, size_y - 1)
        col = np.clip(np.rint(path[:, 1]).astype(int), 0, size_x - 1)
        mean_embedding = embedding_np[:, row, col].mean(axis=1)
        embedding_means.append(mean_embedding)
        mean_probability = float(probability[row, col].mean())
        # The variance head is trained in units of the *truncated normalized
        # distance target* d / distance_truncation.  Convert it back to chart
        # length; multiplying only by the output-pixel size would silently make
        # uncertainty resolution dependent.
        pixel_scale = 1.0 / frame.scale_px
        source_sigma = distance_truncation * torch.exp(
            0.5
            * torch.tensor(
                np.clip(log_variance_np[row, col], -8.0, 6.0), dtype=points.dtype
            )
        )
        sigma = _resample_scalar_field(source_points, source_sigma, points).clamp_min(
            pixel_scale
        )
        candidates.append((points, mean_probability, sigma, mean_embedding))
    if not candidates:
        return []
    ring_ids = _cluster_embeddings(embedding_means, embedding_similarity)
    arcs = []
    for index, ((points, probability_mean, sigma, mean_embedding), ring_id) in enumerate(
        zip(candidates, ring_ids)
    ):
        tangents = torch.nn.functional.normalize(points.roll(-1, 0) - points.roll(1, 0), dim=-1)
        tangents[0] = torch.nn.functional.normalize(points[1] - points[0], dim=-1)
        tangents[-1] = torch.nn.functional.normalize(points[-1] - points[-2], dim=-1)
        reliability = min(max(probability_mean, 0.05), 0.995)
        arc = ArcObservation(
            arc_id=f"predicted:{index}",
            ring_id=ring_id,
            points=points,
            tangents=tangents,
            sigma=sigma.to(points),
            prior_reliability=reliability,
            embedding=torch.tensor(mean_embedding, dtype=points.dtype),
        )
        arc.validate()
        arcs.append(arc)
    return arcs
