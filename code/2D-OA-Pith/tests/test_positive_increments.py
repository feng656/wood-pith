import math

import torch

from oapith.inference.refiner import FarLayerModel, NearGrowthModel, RefinerConfig, pack_arcs
from oapith.types import ArcObservation


def _arc(radius: float, ring_id: int) -> ArcObservation:
    theta = torch.linspace(-1.0, 1.0, 40, dtype=torch.float64)
    points = radius * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    tangents = torch.stack([-torch.sin(theta), torch.cos(theta)], -1)
    return ArcObservation(
        f"ring-{ring_id}", ring_id, points, tangents, torch.tensor(0.003, dtype=torch.float64)
    )


def test_near_growth_parameterization_is_nested_for_all_sampled_angles() -> None:
    config = RefinerConfig(maximum_harmonic=5, disturbance_source_options=(0,))
    data = pack_arcs([_arc(0.4, 9), _arc(0.7, 2), _arc(1.0, 17)], config)
    model = NearGrowthModel(data, torch.zeros(2, dtype=torch.float64), config)
    with torch.no_grad():
        model.gap_shared_coefficients.normal_(0, 1.5)
        model.gap_deviation_coefficients.normal_(0, 1.5)
    theta = torch.linspace(-math.pi, math.pi, 1025, dtype=torch.float64)
    radii = []
    for original_ring in model.ring_order:
        ring_index = torch.full((theta.numel(),), int(original_ring), dtype=torch.long)
        radius, _ = model._radius_and_derivative(theta, ring_index)
        radii.append(radius)
    gaps = torch.stack(radii)[1:] - torch.stack(radii)[:-1]
    assert bool((gaps >= config.nesting_margin - 1e-12).all())


def test_far_offsets_keep_positive_gaps_after_ring_permutation() -> None:
    config = RefinerConfig(disturbance_source_options=(0,))
    data = pack_arcs([_arc(0.9, 17), _arc(0.4, 9), _arc(0.7, 2)], config)
    model = FarLayerModel(
        data,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        0.2,
        config,
    )
    with torch.no_grad():
        model.offset_gap_raw.normal_(0, 2.0)
    offsets_by_internal_ring = model._ring_offsets()
    offsets_in_growth_order = offsets_by_internal_ring[model.ring_order]
    assert bool(
        (
            offsets_in_growth_order[1:] - offsets_in_growth_order[:-1]
            >= config.nesting_margin - 1e-12
        ).all()
    )
