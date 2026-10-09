from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from oapith.inference.candidates import generate_candidates
from oapith.inference.refiner import ProbabilisticGeometryRefiner
from oapith.geometry.curves import estimate_tangents
from oapith.types import ArcObservation


@dataclass
class BootstrapDraw:
    mode: str
    center: list[float] | None
    direction: list[float] | None
    inverse_distance: float | None
    loss: float


def perturb_correlated_arcs(
    arcs: list[ArcObservation],
    *,
    correlation: float = 0.92,
    generator: torch.Generator | None = None,
) -> list[ArcObservation]:
    """Sample a correlated normal annotation realization at the arc-block level."""
    if not 0 <= correlation < 1:
        raise ValueError("correlation must lie in [0,1)")
    output = []
    innovation_scale = (1.0 - correlation**2) ** 0.5
    for arc in arcs:
        cloned = copy.deepcopy(arc)
        count = arc.points.shape[0]
        innovation = torch.randn(
            count,
            generator=generator,
            device=arc.points.device,
            dtype=arc.points.dtype,
        )
        noise = torch.zeros_like(innovation)
        noise[0] = innovation[0]
        for index in range(1, count):
            noise[index] = correlation * noise[index - 1] + innovation_scale * innovation[index]
        sigma = arc.sigma.expand(count) if arc.sigma.ndim == 0 else arc.sigma
        normals = torch.stack([-arc.tangents[:, 1], arc.tangents[:, 0]], -1)
        cloned.points = arc.points + (noise * sigma)[:, None] * normals
        cloned.tangents = estimate_tangents(cloned.points)
        cloned.arc_id = f"{arc.arc_id}:bootstrap"
        output.append(cloned)
    return output


def run_arc_block_bootstrap(
    arcs: list[ArcObservation],
    refiner: ProbabilisticGeometryRefiner,
    *,
    replicates: int = 100,
    seed: int = 2026,
    correlation: float = 0.92,
    neural_candidates: torch.Tensor | None = None,
    state_probabilities: torch.Tensor | None = None,
) -> list[BootstrapDraw]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not arcs:
        raise ValueError("arcs must be non-empty")
    generator = torch.Generator(device=arcs[0].points.device).manual_seed(seed)
    draws = []
    for replicate in range(replicates):
        perturbed = perturb_correlated_arcs(
            arcs, correlation=correlation, generator=generator
        )
        for index, arc in enumerate(perturbed):
            arc.arc_id = f"{arc.arc_id}:{replicate}:{index}"
        candidates = generate_candidates(perturbed, neural_candidates=neural_candidates)
        results, _, _ = refiner.refine(
            perturbed, candidates, state_probabilities=state_probabilities
        )
        best = max(results, key=lambda result: result.evidence_score)
        draws.append(
            BootstrapDraw(
                mode=best.mode.value,
                center=None if best.center is None else best.center.cpu().tolist(),
                direction=None if best.direction is None else best.direction.cpu().tolist(),
                inverse_distance=best.inverse_distance,
                loss=best.loss,
            )
        )
    return draws
