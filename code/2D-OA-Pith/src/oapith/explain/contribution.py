from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from oapith.inference.candidates import generate_candidates
from oapith.inference.posterior import ModePosterior, build_mode_posteriors
from oapith.inference.refiner import ProbabilisticGeometryRefiner
from oapith.types import ArcObservation


@dataclass
class ArcContribution:
    arc_id: str
    reliability: float
    information_gain: float | None
    impact_vector: list[float] | None
    direction_change_radians: float | None
    far_chart_impact: list[float] | None
    predictive_conflict_approx: float
    structural_required: bool
    model_switched: bool
    reduced_mode: str
    reduced_rank: int


def _best_mode(posteriors: list[ModePosterior]) -> ModePosterior:
    return max(posteriors, key=lambda posterior: posterior.weight)


def exact_leave_one_arc_out(
    arcs: list[ArcObservation],
    refiner: ProbabilisticGeometryRefiner,
    full: ModePosterior,
    *,
    neural_candidates: torch.Tensor | None = None,
    state_probabilities: torch.Tensor | None = None,
    arc_preprocessor: Callable[[list[ArcObservation]], list[ArcObservation]] | None = None,
) -> list[ArcContribution]:
    """Re-run weights, shape nuisance, model branches, and Hessian after each deletion."""
    if len(arcs) < 2:
        raise ValueError("leave-one-arc-out requires at least two arcs")
    output: list[ArcContribution] = []
    for removed_index, removed in enumerate(arcs):
        reduced_arcs = [arc for index, arc in enumerate(arcs) if index != removed_index]
        if arc_preprocessor is not None:
            reduced_arcs = arc_preprocessor(reduced_arcs)
        full_start = None
        if full.result.center is not None:
            full_start = full.result.center.detach().reshape(1, 2)
        elif full.result.direction is not None:
            distance = 1.0 / max(full.result.inverse_distance or 0.0, 1e-4)
            full_start = (full.result.direction.detach() * distance).reshape(1, 2)
        proposal = neural_candidates
        if full_start is not None:
            proposal = full_start if proposal is None else torch.cat([full_start.to(proposal), proposal])
        candidates = generate_candidates(reduced_arcs, neural_candidates=proposal)
        results, models, packed = refiner.refine(
            reduced_arcs, candidates, state_probabilities=state_probabilities
        )
        reduced_posteriors = build_mode_posteriors(results, models, packed)
        reduced = _best_mode(reduced_posteriors)
        model_switched = (
            reduced.result.mode is not full.result.mode
            or reduced.result.diagnostics.get("number_disturbance_sources")
            != full.result.diagnostics.get("number_disturbance_sources")
        )
        information_gain: float | None = None
        structural_required = reduced.data.rank < full.data.rank
        full_bounded = full.data.valid and full.data.rank == full.data.dimension
        reduced_bounded = reduced.data.valid and reduced.data.rank == reduced.data.dimension
        if not model_switched and full_bounded and reduced_bounded:
            information_gain = 0.5 * (full.data.pseudo_logdet - reduced.data.pseudo_logdet)
        impact_vector = None
        if full.result.center is not None and reduced.result.center is not None:
            impact_vector = (full.result.center - reduced.result.center).detach().cpu().tolist()
        direction_change = None
        far_chart_impact = None
        if full.result.direction is not None and reduced.result.direction is not None:
            cosine = (full.result.direction * reduced.result.direction).sum().clamp(-1, 1)
            direction_change = float(torch.acos(cosine))
            phi_full = torch.atan2(full.result.direction[1], full.result.direction[0])
            phi_reduced = torch.atan2(reduced.result.direction[1], reduced.result.direction[0])
            delta_phi = torch.atan2(
                torch.sin(phi_full - phi_reduced), torch.cos(phi_full - phi_reduced)
            )
            far_chart_impact = [
                float(delta_phi),
                float((full.result.inverse_distance or 0.0) - (reduced.result.inverse_distance or 0.0)),
            ]
        # Laplace/BIC evidence difference approximates -log p(D_i | D_-i). It is
        # reported explicitly as approximate and never used to set the arc weight.
        predictive_conflict = reduced.result.evidence_score - full.result.evidence_score
        if not math.isfinite(predictive_conflict):
            predictive_conflict = math.inf
        output.append(
            ArcContribution(
                arc_id=removed.arc_id,
                reliability=full.result.reliabilities.get(removed.arc_id, float("nan")),
                information_gain=information_gain,
                impact_vector=impact_vector,
                direction_change_radians=direction_change,
                far_chart_impact=far_chart_impact,
                predictive_conflict_approx=predictive_conflict,
                structural_required=structural_required,
                model_switched=model_switched,
                reduced_mode=reduced.result.mode.value,
                reduced_rank=reduced.data.rank,
            )
        )
    return output
