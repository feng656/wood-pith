from __future__ import annotations

import numpy as np

from racpith.numerics import (
    EvidenceView,
    VarProProblem,
    _compact_components,
    common_center_seed,
)


def concentric_fixture(duplicate: int = 1) -> tuple[EvidenceView, np.ndarray]:
    center = np.asarray([1.7, -0.8], dtype=np.float64)
    points: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    rings: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for ring_index, radius in enumerate((2.0, 2.7, 3.5)):
        angle = np.linspace(0.15, 1.35, 81)
        radial = np.column_stack([np.cos(angle), np.sin(angle)])
        ring_points = center + radius * radial
        ring_tangent = np.column_stack([-np.sin(angle), np.cos(angle)])
        points.append(np.repeat(ring_points, duplicate, axis=0))
        tangents.append(np.repeat(ring_tangent, duplicate, axis=0))
        rings.append(np.full(len(angle) * duplicate, ring_index, dtype=np.int64))
        weights.append(np.full(len(angle) * duplicate, 1.0 / (3 * len(angle) * duplicate)))
    xy = np.vstack(points)
    tangent = np.vstack(tangents)
    ring_index = np.concatenate(rings)
    mass = np.concatenate(weights)
    sigma_x = np.full(len(xy), 0.002)
    sigma_alg = np.full(len(xy), 0.01)
    return (
        EvidenceView(
            points=xy,
            tangents=tangent,
            sigma_x=sigma_x,
            sigma_alg=sigma_alg,
            weight=mass,
            ring_index=ring_index,
        ),
        center,
    )


def test_radius_free_identity_recovers_exact_common_center() -> None:
    evidence, expected = concentric_fixture()
    seed = common_center_seed(
        evidence,
        loss="pseudo_huber",
        delta=1.5,
        max_iterations=100,
        step_tolerance=1e-12,
    )
    assert np.linalg.norm(seed.center - expected) < 1e-9


def test_point_duplication_does_not_change_evidence_or_seed() -> None:
    original, _ = concentric_fixture(duplicate=1)
    duplicated, _ = concentric_fixture(duplicate=4)
    kwargs = dict(loss="pseudo_huber", delta=1.5, max_iterations=100, step_tolerance=1e-12)
    first = common_center_seed(original, **kwargs)
    second = common_center_seed(duplicated, **kwargs)
    assert np.allclose(first.center, second.center, atol=1e-10, rtol=1e-10)
    assert np.isclose(first.objective, second.objective, atol=1e-10, rtol=1e-10)


def test_varpro_envelope_gradient_matches_finite_difference() -> None:
    evidence, expected = concentric_fixture()
    problem = VarProProblem(
        evidence,
        loss="pseudo_huber",
        delta=1.5,
        radius_tolerance=1e-12,
        radius_max_iterations=100,
    )
    center = expected + np.asarray([0.12, -0.07])
    analytic = problem.evaluate(center).gradient
    step = 1e-6
    finite = np.empty(2)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        finite[axis] = (problem.objective(center + offset) - problem.objective(center - offset)) / (
            2 * step
        )
    assert np.allclose(analytic, finite, atol=2e-5, rtol=2e-4)


def test_analytic_far_objective_is_large_radius_limit() -> None:
    evidence, _ = concentric_fixture()
    problem = VarProProblem(
        evidence,
        loss="pseudo_huber",
        delta=1.5,
        radius_tolerance=1e-12,
        radius_max_iterations=100,
    )
    direction = np.asarray([0.6, 0.8])
    analytic = problem.far_objective(direction)
    approximate = problem.objective(1e6 * direction)
    assert np.isclose(analytic, approximate, atol=1e-4, rtol=1e-4)


def test_profiled_radii_obey_first_order_equation() -> None:
    evidence, expected = concentric_fixture()
    problem = VarProProblem(
        evidence,
        loss="pseudo_huber",
        delta=1.5,
        radius_tolerance=1e-12,
        radius_max_iterations=100,
    )
    evaluation = problem.evaluate(expected + np.asarray([0.05, 0.03]))
    distances = np.linalg.norm(evidence.points - (expected + np.asarray([0.05, 0.03])), axis=1)
    for ring, radius in evaluation.radii.items():
        mask = evidence.ring_index == ring
        z = (distances[mask] - radius) / evidence.sigma_x[mask]
        psi = z / np.hypot(1.0, z / 1.5)
        score = np.sum(evidence.weight[mask] * psi / evidence.sigma_x[mask])
        assert abs(score) < 1e-6


def test_compact_topology_identifies_antipodes_only_at_infinity() -> None:
    directions = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    radii = (0.0, 1.0, None)

    infinite = np.full((8, 3), 10.0)
    infinite[0, -1] = 0.0
    infinite[4, -1] = 0.0
    infinite_components = _compact_components(
        infinite, directions, radii, threshold=1.0
    )
    assert len(infinite_components) == 1
    assert infinite_components[0].touches_infinity

    finite = np.full((8, 3), 10.0)
    finite[0, 1] = 0.0
    finite[4, 1] = 0.0
    finite_components = _compact_components(finite, directions, radii, threshold=1.0)
    assert len(finite_components) == 2
    assert not any(component.touches_infinity for component in finite_components)
