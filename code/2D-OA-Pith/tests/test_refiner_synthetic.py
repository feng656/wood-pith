import math
import types

import pytest
import torch

from oapith.inference.refiner import (
    ProbabilisticGeometryRefiner,
    RefinerConfig,
    RefinementMode,
    RefinementResult,
)
from oapith.types import ArcObservation, PithState


def _arc(center: torch.Tensor, radius: float, start: float, stop: float, ring_id: int, name: str):
    theta = torch.linspace(start, stop, 80, dtype=torch.float64)
    points = center + radius * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    tangents = torch.stack([-torch.sin(theta), torch.cos(theta)], -1)
    return ArcObservation(
        name,
        ring_id,
        points,
        tangents,
        torch.tensor(0.003, dtype=torch.float64),
    )


@pytest.mark.slow
def test_near_refiner_can_keep_a_center_outside_the_canvas() -> None:
    truth = torch.tensor([1.45, -0.35], dtype=torch.float64)
    arcs = [
        _arc(truth, 1.1, 2.4, 3.7, 0, "a"),
        _arc(truth, 1.3, 2.3, 3.8, 1, "b"),
        _arc(truth, 1.5, 2.2, 3.9, 2, "c"),
    ]
    config = RefinerConfig(
        maximum_harmonic=3,
        warmup_steps=8,
        stage_steps=8,
        em_rounds=1,
        disturbance_source_options=(0,),
        max_modes=1,
    )
    results, _, _ = ProbabilisticGeometryRefiner(config).refine(
        arcs, [truth + torch.tensor([0.1, -0.1])], modes=(RefinementMode.NEAR,)
    )
    assert results[0].center is not None
    assert torch.linalg.vector_norm(results[0].center - truth) < 0.15
    assert results[0].center[0] > 1.0


def test_mode_quota_never_overfills_and_prioritizes_chart_evidence() -> None:
    arc = _arc(torch.zeros(2, dtype=torch.float64), 0.8, -0.5, 0.5, 0, "quota")
    refiner = ProbabilisticGeometryRefiner(
        RefinerConfig(disturbance_source_options=(0,), max_modes=2)
    )
    evidence = {
        RefinementMode.NEAR: 10.0,
        RefinementMode.FAR: 9.0,
        RefinementMode.INFINITY: 8.0,
    }
    states = {
        RefinementMode.NEAR: PithState.NEAR,
        RefinementMode.FAR: PithState.FAR,
        RefinementMode.INFINITY: PithState.INFINITY,
    }

    def fake_fit(self, data, start, mode, number_sources):
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
        result = RefinementResult(
            mode=mode,
            state=states[mode],
            center=torch.zeros(2, dtype=torch.float64)
            if mode is RefinementMode.NEAR
            else None,
            direction=None if mode is RefinementMode.NEAR else direction,
            inverse_distance=0.2 if mode is RefinementMode.FAR else None,
            loss=-evidence[mode],
            reliabilities={"quota": 0.9},
            model_state={},
            evidence_score=evidence[mode],
        )
        return object(), result

    refiner._fit_one = types.MethodType(fake_fit, refiner)
    results, _, _ = refiner.refine(
        [arc],
        [torch.zeros(2, dtype=torch.float64), torch.tensor([2.0, 0.0], dtype=torch.float64)],
    )
    assert len(results) == 2
    assert [result.mode for result in results] == [RefinementMode.NEAR, RefinementMode.FAR]


def test_refiner_rejects_duplicate_chart_requests() -> None:
    arc = _arc(torch.zeros(2, dtype=torch.float64), 0.8, -0.5, 0.5, 0, "duplicate")
    refiner = ProbabilisticGeometryRefiner(
        RefinerConfig(disturbance_source_options=(0,))
    )
    with pytest.raises(ValueError, match="unique charts"):
        refiner.refine(
            [arc],
            [torch.zeros(2, dtype=torch.float64)],
            modes=(RefinementMode.NEAR, RefinementMode.NEAR),
        )
