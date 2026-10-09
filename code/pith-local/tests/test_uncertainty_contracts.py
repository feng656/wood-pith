from __future__ import annotations

import numpy as np

from racpith.contracts import EvidenceBundle
from racpith.contribution import perturb_evidence
from racpith.uncertainty import ring_bootstrap_bundle


def _bundle() -> EvidenceBundle:
    angles = np.linspace(0.1, 1.2, 12)
    points = []
    tangents = []
    for radius in (1.0, 1.4, 1.9):
        radial = np.column_stack([np.cos(angles), np.sin(angles)])
        points.append(radius * radial)
        tangents.append(np.column_stack([-np.sin(angles), np.cos(angles)]))
    xy = np.vstack(points)
    tangent = np.vstack(tangents)
    ring_index = np.repeat(np.arange(3), len(angles))
    arc_index = ring_index.copy()
    scale = float(np.hypot(64, 64))
    return EvidenceBundle(
        crop_id="fixture",
        tree_id="T0_B1_N1",
        section_id="T0_B1_N1_A",
        points_norm=xy,
        points_crop_px=xy * scale + 32,
        tangents=tangent,
        sigma_x_norm=np.full(len(xy), 0.01),
        sigma_alg_norm2=np.full(len(xy), 0.02),
        base_weight=np.full(len(xy), 1.0 / len(xy)),
        ring_index=ring_index,
        arc_index=arc_index,
        source_s_px=np.tile(np.arange(len(angles), dtype=float), 3),
        arc_fraction=np.tile(np.linspace(0.0, 1.0, len(angles)), 3),
        quality=np.ones(len(xy)),
        ring_ids=("r0", "r1", "r2"),
        arc_ids=("a0", "a1", "a2"),
        crop_origin_source_px=(10.0, 20.0),
        crop_size_px=(64, 64),
        normalization_scale_px=scale,
        metadata={"ring_order_reliable": True},
    )


def test_parent_ring_bootstrap_reindexes_duplicates_and_equalizes_draw_mass() -> None:
    boot = ring_bootstrap_bundle(_bundle(), np.asarray([2, 0, 2]))
    boot.validate()
    assert len(boot.ring_ids) == 3
    assert len(set(boot.ring_ids)) == 3
    mass = np.bincount(boot.ring_index, weights=boot.base_weight)
    assert np.allclose(mass, np.full(3, 1.0 / 3.0))
    assert boot.metadata["bootstrap_source_ring_ids"] == ["r2", "r0", "r2"]
    assert boot.metadata["ring_order_reliable"] is False
    assert boot.metadata["gt_available_to_estimator"] is False


def test_correlated_perturbation_preserves_coordinate_contract() -> None:
    perturbed = perturb_evidence(_bundle(), np.random.default_rng(17), 0.25)
    perturbed.validate()
    width, height = perturbed.crop_size_px
    expected = (
        perturbed.points_crop_px - np.asarray([width / 2.0, height / 2.0])
    ) / perturbed.normalization_scale_px
    assert np.allclose(perturbed.points_norm, expected)
