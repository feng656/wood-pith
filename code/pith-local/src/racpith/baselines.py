from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import EvidenceBundle
from .estimator import RacPithEstimator
from .numerics import (
    EvidenceView,
    VarProProblem,
    common_center_seed,
    optimise_varpro,
    tangent_intersection_seed,
)


BASELINE_METHODS = (
    "B0_independent_circle_median",
    "B1_tangent_robust_seed",
    "B2_convex_radius_free_seed",
    "B3_l2_common_center_varpro",
    "B4_robust_common_center_varpro",
    "B5_robust_varpro_far_profile_state",
)


def _view(bundle: EvidenceBundle) -> EvidenceView:
    return EvidenceView(
        points=bundle.points_norm,
        tangents=bundle.tangents,
        sigma_x=bundle.sigma_x_norm,
        sigma_alg=bundle.sigma_alg_norm2,
        weight=bundle.base_weight,
        ring_index=bundle.ring_index,
    )


def _weighted_circle(points: np.ndarray, weight: np.ndarray) -> np.ndarray:
    design = np.column_stack([2.0 * points, np.ones(len(points))])
    target = np.sum(points * points, axis=1)
    root = np.sqrt(weight / weight.sum())
    solution, _, rank, _ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
    if rank < 3 or not np.all(np.isfinite(solution[:2])):
        raise ValueError("independent ring circle fit is rank deficient")
    return solution[:2]


def _independent_radius_residual_objective(
    view: EvidenceView, center: np.ndarray
) -> float:
    """Score a finite baseline without constructing the robust VarPro object.

    B0--B2 are deliberately seed/comparator methods.  Their diagnostic
    objective must not make them depend on the shared robust VarPro
    construction used by B4/B5; a failure in that construction is local to
    the methods that actually need it.
    """

    distances = np.linalg.norm(view.points - np.asarray(center, dtype=float), axis=1)
    if distances.shape != view.weight.shape or not np.all(np.isfinite(distances)):
        raise ValueError("baseline centre produced non-finite radial distances")
    value = 0.0
    for ring in view.active_rings:
        mask = view.ring_index == ring
        weights = view.weight[mask]
        mass = float(weights.sum())
        if mass <= 0.0:
            raise ValueError("baseline ring has non-positive weight")
        radius = float(np.sum(weights * distances[mask]) / mass)
        value += float(np.sum(weights * np.square(distances[mask] - radius)))
    if not np.isfinite(value):
        raise ValueError("baseline diagnostic objective is non-finite")
    return value


def _finite_record(
    method: str,
    center: np.ndarray,
    objective: float,
    converged: bool,
    config_hash: str,
) -> dict[str, Any]:
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError(f"{method} returned a non-finite two-dimensional center")
    if not np.isfinite(objective):
        raise ValueError(f"{method} returned a non-finite objective")
    return {
        "schema_version": "racpith.baseline.v1",
        "method": method,
        "raw_center_norm": [float(center[0]), float(center[1])],
        "objective_under_method": float(objective),
        "converged": bool(converged),
        "state": "POINT",
        "state_policy": "ALWAYS_FINITE_BASELINE_NO_IDENTIFIABILITY_GUARD",
        "search_adequate": False,
        "config_hash": config_hash,
        "status": "FINISHED",
    }


def _failure_record(
    method: str,
    config_hash: str,
    exc: Exception,
) -> dict[str, Any]:
    """Represent one failed comparator without suppressing the other methods."""

    return {
        "schema_version": "racpith.baseline.v1",
        "method": method,
        "raw_center_norm": None,
        "objective_under_method": None,
        "converged": False,
        "state": "REJECT",
        "state_policy": "METHOD_LOCAL_FAILURE",
        "search_adequate": False,
        "config_hash": config_hash,
        "status": "CRASH",
        "reason": f"{type(exc).__name__}: {exc}",
    }


def run_geometric_baselines(
    bundle: EvidenceBundle, estimator: RacPithEstimator
) -> list[dict[str, Any]]:
    """Run every registered comparator with method-local failure isolation.

    Evidence decoding/validation is deliberately outside this isolation boundary,
    because all six methods share that input.  Once a valid bundle exists, a
    rank-deficient circle, failed tangent seed, or optimiser exception is recorded
    only for the affected comparator; B5 and the independent methods still run.
    """

    bundle.validate()
    view = _view(bundle)
    solver = estimator.solver
    config_hash = estimator.config.sha256
    records: dict[str, dict[str, Any]] = {}

    robust_problem: VarProProblem | None = None
    robust_problem_error: Exception | None = None
    try:
        robust_problem = VarProProblem(
            view,
            loss=estimator.loss,
            delta=estimator.delta,
            radius_tolerance=float(solver["radius_tol"]),
            radius_max_iterations=int(solver["radius_max_iter"]),
        )
    except Exception as exc:  # converted to one record per dependent method below
        robust_problem_error = exc

    common = None
    common_error: Exception | None = None
    try:
        common = common_center_seed(
            view,
            loss=estimator.loss,
            delta=estimator.delta,
            max_iterations=int(solver["irls_max_iter"]),
            step_tolerance=float(solver["irls_step_tol"]),
        )
    except Exception as exc:
        common_error = exc

    method = "B0_independent_circle_median"
    try:
        circle_centers = []
        for ring in view.active_rings:
            mask = view.ring_index == ring
            circle_centers.append(_weighted_circle(view.points[mask], view.weight[mask]))
        if not circle_centers:
            raise ValueError("no active ring is available for independent circle fits")
        center = np.median(np.stack(circle_centers), axis=0)
        records[method] = _finite_record(
            method,
            center,
            _independent_radius_residual_objective(view, center),
            True,
            config_hash,
        )
    except Exception as exc:
        records[method] = _failure_record(method, config_hash, exc)

    method = "B1_tangent_robust_seed"
    try:
        tangent = tangent_intersection_seed(
            view,
            loss=estimator.loss,
            delta=estimator.delta,
            max_iterations=int(solver["irls_max_iter"]),
            step_tolerance=float(solver["irls_step_tol"]),
        )
        records[method] = _finite_record(
            method,
            tangent.center,
            _independent_radius_residual_objective(view, tangent.center),
            tangent.converged,
            config_hash,
        )
    except Exception as exc:
        records[method] = _failure_record(method, config_hash, exc)

    method = "B2_convex_radius_free_seed"
    try:
        if common is None:
            raise RuntimeError("common-center seed failed") from common_error
        records[method] = _finite_record(
            method,
            common.center,
            _independent_radius_residual_objective(view, common.center),
            common.converged,
            config_hash,
        )
    except Exception as exc:
        records[method] = _failure_record(method, config_hash, exc)

    method = "B3_l2_common_center_varpro"
    try:
        if common is None:
            raise RuntimeError("common-center seed failed") from common_error
        l2_problem = VarProProblem(
            view,
            loss="l2",
            delta=estimator.delta,
            radius_tolerance=float(solver["radius_tol"]),
            radius_max_iterations=int(solver["radius_max_iter"]),
        )
        fit = optimise_varpro(
            l2_problem,
            common.center,
            max_iterations=int(solver["outer_max_iter"]),
            gradient_tolerance=float(solver["outer_grad_tol"]),
            acceptable_gradient_tolerance=float(solver["acceptable_grad_tol"]),
            numerical_radius_limit=float(solver["numerical_radius_limit"]),
        )
        records[method] = _finite_record(
            method,
            fit.center,
            fit.objective,
            fit.converged and not fit.escaped,
            config_hash,
        )
    except Exception as exc:
        records[method] = _failure_record(method, config_hash, exc)

    method = "B4_robust_common_center_varpro"
    try:
        if common is None:
            raise RuntimeError("common-center seed failed") from common_error
        if robust_problem is None:
            raise RuntimeError("robust objective construction failed") from robust_problem_error
        fit = optimise_varpro(
            robust_problem,
            common.center,
            max_iterations=int(solver["outer_max_iter"]),
            gradient_tolerance=float(solver["outer_grad_tol"]),
            acceptable_gradient_tolerance=float(solver["acceptable_grad_tol"]),
            numerical_radius_limit=float(solver["numerical_radius_limit"]),
        )
        records[method] = _finite_record(
            method,
            fit.center,
            fit.objective,
            fit.converged and not fit.escaped,
            config_hash,
        )
    except Exception as exc:
        records[method] = _failure_record(method, config_hash, exc)

    method = "B5_robust_varpro_far_profile_state"
    try:
        full = estimator.fit(bundle)
        records[method] = {
            "schema_version": "racpith.baseline.v1",
            "method": method,
            "raw_center_norm": full.raw_center_norm,
            "objective_under_method": full.objective,
            "converged": full.raw_center_norm is not None,
            "state": full.state.value,
            "state_policy": "RACPITH_IDENTIFIABILITY_STATE",
            "search_adequate": full.search_adequate,
            "config_hash": config_hash,
            "status": "FINISHED",
        }
    except Exception as exc:
        records[method] = _failure_record(method, config_hash, exc)

    if set(records) != set(BASELINE_METHODS):
        raise AssertionError("baseline implementation did not cover the registered methods")
    return [records[method] for method in BASELINE_METHODS]
