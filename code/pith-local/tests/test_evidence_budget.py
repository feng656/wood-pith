from __future__ import annotations

from pathlib import Path

import numpy as np

from racpith.config import load_config
from racpith.contracts import EvidenceBundle
from racpith.estimator import RacPithEstimator
from racpith.groups import build_deletion_groups, validate_partition_mass
from racpith.numerics import EvidenceView, VarProProblem
from racpith.provenance import sha256_file


def fixture_bundle() -> EvidenceBundle:
    n_per_ring = 20
    ring_index = np.repeat(np.arange(3), n_per_ring)
    arc_index = ring_index.copy()
    angle = np.tile(np.linspace(0.0, 1.0, n_per_ring), 3)
    points = np.column_stack([np.cos(angle), np.sin(angle)])
    tangent = np.column_stack([-np.sin(angle), np.cos(angle)])
    scale = float(np.hypot(256, 256))
    return EvidenceBundle(
        crop_id="fixture",
        tree_id="tree",
        section_id="section",
        points_norm=points,
        points_crop_px=points * scale + 128,
        tangents=tangent,
        sigma_x_norm=np.full(len(points), 0.01),
        sigma_alg_norm2=np.full(len(points), 0.02),
        base_weight=np.full(len(points), 1.0 / len(points)),
        ring_index=ring_index,
        arc_index=arc_index,
        source_s_px=angle * 100,
        arc_fraction=angle,
        quality=np.ones(len(points)),
        ring_ids=("r0", "r1", "r2"),
        arc_ids=("a0", "a1", "a2"),
        crop_origin_source_px=(0.0, 0.0),
        crop_size_px=(256, 256),
        normalization_scale_px=scale,
    )


def test_registered_partitions_cover_each_node_once() -> None:
    bundle = fixture_bundle()
    groups = build_deletion_groups(
        bundle,
        {
            "levels": ["ring", "arc", "subarc"],
            "primary_subarc_fraction": 0.1,
            "audit_subarc_fractions": [0.05, 0.1, 0.2],
            "audit_phases": [0.0, 0.5],
        },
        include_audit_partitions=True,
    )
    assert validate_partition_mass(bundle, groups) == []


def test_deletion_does_not_renormalize_sibling_mass() -> None:
    bundle = fixture_bundle()
    groups = build_deletion_groups(
        bundle,
        {
            "levels": ["subarc"],
            "primary_subarc_fraction": 0.1,
            "audit_subarc_fractions": [0.1],
            "audit_phases": [0.0],
        },
    )
    group = groups[0]
    remaining = ~group.remove_mask
    assert np.isclose(
        bundle.base_weight[remaining].sum(), bundle.base_weight.sum() - group.removed_mass
    )
    assert np.all(bundle.base_weight[remaining] == fixture_bundle().base_weight[remaining])


def test_posthoc_support_scorer_is_the_same_profiled_objective() -> None:
    bundle = fixture_bundle()
    frozen = load_config(Path(__file__).resolve().parents[1] / "configs" / "racpith_v1.json")
    estimator = RacPithEstimator(frozen, run_id="test-support-score")
    center = np.asarray([0.12, -0.08], dtype=np.float64)
    view = EvidenceView(
        points=bundle.points_norm,
        tangents=bundle.tangents,
        sigma_x=bundle.sigma_x_norm,
        sigma_alg=bundle.sigma_alg_norm2,
        weight=bundle.base_weight,
        ring_index=bundle.ring_index,
    )
    direct = VarProProblem(
        view,
        loss=estimator.loss,
        delta=estimator.delta,
        radius_tolerance=float(estimator.solver["radius_tol"]),
        radius_max_iterations=int(estimator.solver["radius_max_iter"]),
    ).objective(center)
    assert np.isclose(
        estimator.objective_at_center(bundle, center), direct, rtol=0.0, atol=1e-12
    )


def test_evidence_npz_is_byte_deterministic_and_closes_archive(tmp_path) -> None:
    bundle = fixture_bundle()
    first = tmp_path / "first"
    second = tmp_path / "second"
    bundle.save(first / "evidence.npz", first / "evidence.json")
    bundle.save(second / "evidence.npz", second / "evidence.json")
    assert sha256_file(first / "evidence.npz") == sha256_file(second / "evidence.npz")
    restored = EvidenceBundle.load(first / "evidence.json")
    assert np.array_equal(restored.points_norm, bundle.points_norm)
