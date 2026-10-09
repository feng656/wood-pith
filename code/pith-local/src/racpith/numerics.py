"""Numerical core for RAC-Pith v2.

The routines in this module know nothing about pith ground truth.  They operate
on an immutable, candidate-independent quadrature measure supplied by the
estimator.  In particular, deleting evidence means removing rows; the retained
weights are never renormalised.

All geometry is evaluated in normalised crop coordinates and in float64.  The
finite optimisation is deliberately unbounded.  Infinity is not represented by
a large finite centre: it has its own analytic, per-ring profiled objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar


LossName = Literal["pseudo_huber", "l2"]

_FLOAT_EPS = np.finfo(np.float64).eps
_TINY = np.finfo(np.float64).tiny
_HUGE_OBJECTIVE = 1.0e200


def _as_float64(value: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def canonical_axis(vector: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a deterministic unit representative of an unoriented 2-D axis."""

    axis = _as_float64(vector).reshape(2)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= _TINY:
        return np.array([1.0, 0.0], dtype=np.float64)
    axis = axis / norm
    if axis[0] < 0.0 or (abs(axis[0]) <= 32.0 * _FLOAT_EPS and axis[1] < 0.0):
        axis = -axis
    return axis


def projective_angle_distance(phi_a: float, phi_b: float) -> float:
    """Angular distance on an unoriented axis, hence periodic with period pi."""

    delta = abs((float(phi_a) - float(phi_b)) % pi)
    return float(min(delta, pi - delta))


def loss_rho(z: np.ndarray, loss: LossName, delta: float) -> np.ndarray:
    z64 = _as_float64(z)
    if loss == "l2":
        return 0.5 * np.square(z64)
    if loss != "pseudo_huber":
        raise ValueError(f"unsupported loss: {loss!r}")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("pseudo-Huber delta must be finite and positive")
    scaled = z64 / float(delta)
    return float(delta) ** 2 * (np.hypot(1.0, scaled) - 1.0)


def loss_psi(z: np.ndarray, loss: LossName, delta: float) -> np.ndarray:
    z64 = _as_float64(z)
    if loss == "l2":
        return z64
    if loss != "pseudo_huber":
        raise ValueError(f"unsupported loss: {loss!r}")
    scaled = z64 / float(delta)
    return z64 / np.hypot(1.0, scaled)


def irls_multiplier(z: np.ndarray, loss: LossName, delta: float) -> np.ndarray:
    """Return psi(z)/z with its continuous value at zero."""

    z64 = _as_float64(z)
    if loss == "l2":
        return np.ones_like(z64)
    if loss != "pseudo_huber":
        raise ValueError(f"unsupported loss: {loss!r}")
    return 1.0 / np.hypot(1.0, z64 / float(delta))


@dataclass(frozen=True)
class EvidenceView:
    """Active rows of an :class:`EvidenceBundle`, without weight changes."""

    points: np.ndarray
    tangents: np.ndarray
    sigma_x: np.ndarray
    sigma_alg: np.ndarray
    weight: np.ndarray
    ring_index: np.ndarray

    def validate(self) -> None:
        n = int(self.points.shape[0])
        if n == 0:
            raise ValueError("active evidence is empty")
        if self.points.shape != (n, 2) or self.tangents.shape != (n, 2):
            raise ValueError("points and tangents must have shape (N, 2)")
        for name in ("sigma_x", "sigma_alg", "weight", "ring_index"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape (N,)")
        for value in (
            self.points,
            self.tangents,
            self.sigma_x,
            self.sigma_alg,
            self.weight,
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError("active evidence contains NaN or infinity")
        if np.any(self.sigma_x <= 0.0) or np.any(self.sigma_alg <= 0.0):
            raise ValueError("measurement scales must be positive")
        if np.any(self.weight <= 0.0):
            raise ValueError("quadrature weights must be positive")
        if np.any(self.ring_index < 0):
            raise ValueError("ring indices must be non-negative")

    @property
    def active_rings(self) -> np.ndarray:
        return np.unique(self.ring_index)


@dataclass(frozen=True)
class LinearSeedResult:
    center: np.ndarray
    objective: float
    information: np.ndarray
    eigenvalues: np.ndarray
    weak_direction: np.ndarray
    strong_direction: np.ndarray
    condition_ratio: float
    rank: int
    converged: bool
    iterations: int
    reason: str | None


@dataclass(frozen=True)
class VarProEvaluation:
    objective: float
    gradient: np.ndarray
    radii: dict[int, float]


@dataclass(frozen=True)
class FiniteSolution:
    center: np.ndarray
    objective: float
    gradient_norm: float
    radii: dict[int, float]
    converged: bool
    escaped: bool
    iterations: int
    seed: np.ndarray
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "center_norm": [float(self.center[0]), float(self.center[1])],
            "objective": float(self.objective),
            "gradient_norm": float(self.gradient_norm),
            "radii_norm_by_index": {
                str(key): float(value) for key, value in self.radii.items()
            },
            "converged": bool(self.converged),
            "escaped": bool(self.escaped),
            "iterations": int(self.iterations),
            "seed_norm": [float(self.seed[0]), float(self.seed[1])],
            "message": self.message,
        }


@dataclass(frozen=True)
class FarMinimum:
    phi_rad: float
    objective: float

    @property
    def direction(self) -> np.ndarray:
        return canonical_axis(np.array([np.cos(self.phi_rad), np.sin(self.phi_rad)]))

    def as_dict(self) -> dict[str, Any]:
        direction = self.direction
        return {
            "phi_rad": float(self.phi_rad % pi),
            "direction_norm": [float(direction[0]), float(direction[1])],
            "objective": float(self.objective),
        }


@dataclass(frozen=True)
class FarScanResult:
    angles_rad: np.ndarray
    objectives: np.ndarray
    replay_angles_rad: np.ndarray
    replay_objectives: np.ndarray
    minima: tuple[FarMinimum, ...]
    replay_minima: tuple[FarMinimum, ...]
    supported_minima: tuple[FarMinimum, ...]
    best: FarMinimum
    stable: bool
    angle_step_rad: float
    support_axis_spread_rad: float
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "angles_rad": [float(value) for value in self.angles_rad],
            "objective": [float(value) for value in self.objectives],
            "replay_angles_rad": [float(value) for value in self.replay_angles_rad],
            "replay_objective": [float(value) for value in self.replay_objectives],
            "minima": [item.as_dict() for item in self.minima],
            "replay_minima": [item.as_dict() for item in self.replay_minima],
            "supported_minima": [item.as_dict() for item in self.supported_minima],
            "best": self.best.as_dict(),
            "stable": bool(self.stable),
            "angle_step_rad": float(self.angle_step_rad),
            "support_axis_spread_rad": float(self.support_axis_spread_rad),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ProfileComponent:
    start_index: int
    end_index: int
    z_min: float
    z_max: float
    touches_negative_infinity: bool
    touches_positive_infinity: bool
    objective_min: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_index": int(self.start_index),
            "end_index": int(self.end_index),
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
            "touches_negative_infinity": bool(self.touches_negative_infinity),
            "touches_positive_infinity": bool(self.touches_positive_infinity),
            "objective_min": float(self.objective_min),
        }


@dataclass(frozen=True)
class DirectionProfileResult:
    direction: np.ndarray
    strong_direction: np.ndarray
    anchor: np.ndarray
    z: np.ndarray
    objectives: np.ndarray
    eta: np.ndarray
    centers: np.ndarray
    support_threshold: float
    components: tuple[ProfileComponent, ...]
    replay_component_count: int
    stable: bool
    best_objective: float
    finite_seeds: tuple[np.ndarray, ...]
    strong_halfwidth: float
    support_touches_strong_boundary: bool

    def as_dict(self) -> dict[str, Any]:
        centres: list[list[float] | None] = []
        for center in self.centers:
            if np.all(np.isfinite(center)):
                centres.append([float(center[0]), float(center[1])])
            else:
                centres.append(None)
        return {
            "direction_norm": [float(self.direction[0]), float(self.direction[1])],
            "strong_direction_norm": [
                float(self.strong_direction[0]),
                float(self.strong_direction[1]),
            ],
            "anchor_norm": [float(self.anchor[0]), float(self.anchor[1])],
            "z": [float(value) for value in self.z],
            "objective": [float(value) for value in self.objectives],
            "eta_norm": [
                float(value) if np.isfinite(value) else None for value in self.eta
            ],
            "centers_norm": centres,
            "support_threshold": float(self.support_threshold),
            "components": [item.as_dict() for item in self.components],
            "replay_component_count": int(self.replay_component_count),
            "stable": bool(self.stable),
            "best_objective": float(self.best_objective),
            "finite_seeds_norm": [
                [float(seed[0]), float(seed[1])] for seed in self.finite_seeds
            ],
            "strong_halfwidth_norm": float(self.strong_halfwidth),
            "support_touches_strong_boundary": bool(
                self.support_touches_strong_boundary
            ),
        }


@dataclass(frozen=True)
class CompactComponent:
    cell_count: int
    touches_origin: bool
    touches_infinity: bool
    touches_radius_cap: bool
    radius_min: float
    radius_max: float | None
    theta_min_rad: float
    theta_max_rad: float
    objective_min: float
    best_center: np.ndarray | None
    best_theta_rad: float
    finite_span_norm: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_count": int(self.cell_count),
            "touches_origin": bool(self.touches_origin),
            "touches_infinity": bool(self.touches_infinity),
            "touches_radius_cap": bool(self.touches_radius_cap),
            "radius_min_norm": float(self.radius_min),
            "radius_max_norm": (
                float(self.radius_max) if self.radius_max is not None else None
            ),
            "theta_min_rad": float(self.theta_min_rad),
            "theta_max_rad": float(self.theta_max_rad),
            "objective_min": float(self.objective_min),
            "best_theta_rad": float(self.best_theta_rad),
            "finite_span_norm": (
                float(self.finite_span_norm)
                if self.finite_span_norm is not None
                else None
            ),
            "best_center_norm": (
                [float(self.best_center[0]), float(self.best_center[1])]
                if self.best_center is not None
                else None
            ),
        }


@dataclass(frozen=True)
class CompactAuditResult:
    directions_rad: np.ndarray
    radii: tuple[float | None, ...]
    objectives: np.ndarray
    support_threshold: float
    components: tuple[CompactComponent, ...]
    replay_component_count: int
    stable: bool
    best_objective: float
    best_center: np.ndarray | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameterization": "direction_and_inverse_distance",
            "infinity_topology": "projective_axis_antipodes_identified",
            "directions_rad": [float(value) for value in self.directions_rad],
            "radii_norm": [
                float(value) if value is not None else None for value in self.radii
            ],
            "inverse_distance": [
                1.0 / (1.0 + float(value)) if value is not None else 0.0
                for value in self.radii
            ],
            "objective": [
                [float(value) for value in row] for row in self.objectives
            ],
            "support_threshold": float(self.support_threshold),
            "components": [item.as_dict() for item in self.components],
            "replay_component_count": int(self.replay_component_count),
            "stable": bool(self.stable),
            "best_objective": float(self.best_objective),
            "best_center_norm": (
                [float(self.best_center[0]), float(self.best_center[1])]
                if self.best_center is not None
                else None
            ),
        }


def robust_location(
    values: np.ndarray,
    sigma: np.ndarray,
    weight: np.ndarray,
    *,
    loss: LossName,
    delta: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Strictly solve the one-dimensional profiled location problem.

    For pseudo-Huber this solves the monotone first-order equation with a
    bracketed root.  It intentionally does not substitute a median or a mean.
    """

    values64 = _as_float64(values).reshape(-1)
    sigma64 = _as_float64(sigma).reshape(-1)
    weight64 = _as_float64(weight).reshape(-1)
    if values64.size == 0 or sigma64.shape != values64.shape or weight64.shape != values64.shape:
        raise ValueError("robust_location arrays must be non-empty and have equal shape")
    if not (
        np.all(np.isfinite(values64))
        and np.all(np.isfinite(sigma64))
        and np.all(np.isfinite(weight64))
    ):
        raise ValueError("robust_location received a non-finite value")
    if np.any(sigma64 <= 0.0) or np.any(weight64 <= 0.0):
        raise ValueError("robust_location scales and weights must be positive")

    lower = float(np.min(values64))
    upper = float(np.max(values64))
    if upper - lower <= max(float(tolerance), 32.0 * _FLOAT_EPS * max(1.0, abs(lower))):
        return 0.5 * (lower + upper)
    if loss == "l2":
        precision_weight = weight64 / np.square(sigma64)
        return float(np.dot(precision_weight, values64) / np.sum(precision_weight))

    def score(location: float) -> float:
        standardised = (values64 - float(location)) / sigma64
        return float(
            np.sum(weight64 * loss_psi(standardised, loss, delta) / sigma64)
        )

    score_lower = score(lower)
    score_upper = score(upper)
    scale = max(1.0, abs(score_lower), abs(score_upper))
    if score_lower < -1e-12 * scale or score_upper > 1e-12 * scale:
        raise ArithmeticError("robust location root is not bracketed")
    if abs(score_lower) <= 1e-14 * scale:
        return lower
    if abs(score_upper) <= 1e-14 * scale:
        return upper
    rtol = max(4.0 * _FLOAT_EPS, float(tolerance))
    return float(
        brentq(
            score,
            lower,
            upper,
            xtol=max(float(tolerance), 4.0 * _FLOAT_EPS),
            rtol=rtol,
            maxiter=int(max_iterations),
        )
    )


def _linear_objective(
    design: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    weight: np.ndarray,
    center: np.ndarray,
    loss: LossName,
    delta: float,
) -> float:
    residual = (design @ center - target) / sigma
    value = float(np.sum(weight * loss_rho(residual, loss, delta)))
    return value if np.isfinite(value) else _HUGE_OBJECTIVE


def _solve_psd(system: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, int]:
    symmetric = 0.5 * (system + system.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    maximum = float(np.max(eigenvalues, initial=0.0))
    cutoff = max(1e-14, maximum * 1e-12)
    active = eigenvalues > cutoff
    rank = int(np.count_nonzero(active))
    inverse = np.zeros_like(eigenvalues)
    inverse[active] = 1.0 / eigenvalues[active]
    solution = eigenvectors @ (inverse * (eigenvectors.T @ rhs))
    return solution, rank


def solve_convex_irls(
    design: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    weight: np.ndarray,
    *,
    loss: LossName,
    delta: float,
    max_iterations: int,
    step_tolerance: float,
) -> LinearSeedResult:
    design64 = _as_float64(design)
    target64 = _as_float64(target).reshape(-1)
    sigma64 = _as_float64(sigma).reshape(-1)
    weight64 = _as_float64(weight).reshape(-1)
    if design64.ndim != 2 or design64.shape[1] != 2:
        raise ValueError("linear seed design must have shape (N, 2)")
    if not (
        design64.shape[0]
        == target64.size
        == sigma64.size
        == weight64.size
    ):
        raise ValueError("linear seed arrays have incompatible lengths")
    if design64.shape[0] < 2:
        raise ValueError("at least two constraints are required")

    base_precision = weight64 / np.square(sigma64)
    initial_system = design64.T @ (base_precision[:, None] * design64)
    initial_rhs = design64.T @ (base_precision * target64)
    center, _ = _solve_psd(initial_system, initial_rhs)
    previous = _linear_objective(
        design64, target64, sigma64, weight64, center, loss, delta
    )
    converged = False
    reason: str | None = None
    iterations = 0

    for iteration in range(1, int(max_iterations) + 1):
        iterations = iteration
        residual = (design64 @ center - target64) / sigma64
        robust_weight = irls_multiplier(residual, loss, delta)
        precision = weight64 * robust_weight / np.square(sigma64)
        system = design64.T @ (precision[:, None] * design64)
        rhs = design64.T @ (precision * target64)
        proposal, _ = _solve_psd(system, rhs)
        step = proposal - center
        if not np.all(np.isfinite(step)):
            reason = "NONFINITE_IRLS_STEP"
            break

        multiplier = 1.0
        accepted = False
        trial = center
        trial_value = previous
        while multiplier >= 2.0 ** -24:
            candidate = center + multiplier * step
            candidate_value = _linear_objective(
                design64,
                target64,
                sigma64,
                weight64,
                candidate,
                loss,
                delta,
            )
            if candidate_value <= previous + 1e-13 * max(1.0, abs(previous)):
                trial = candidate
                trial_value = candidate_value
                accepted = True
                break
            multiplier *= 0.5
        if not accepted:
            reason = "IRLS_LINE_SEARCH_FAILED"
            break
        center = trial
        step_norm = float(np.linalg.norm(multiplier * step))
        previous = trial_value
        if step_norm <= float(step_tolerance) * (1.0 + float(np.linalg.norm(center))):
            converged = True
            break
    else:
        reason = "IRLS_MAX_ITERATIONS"

    residual = (design64 @ center - target64) / sigma64
    robust_weight = irls_multiplier(residual, loss, delta)
    precision = weight64 * robust_weight / np.square(sigma64)
    information = design64.T @ (precision[:, None] * design64)
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    largest = float(eigenvalues[-1])
    cutoff = max(1e-14, largest * 1e-12)
    rank = int(np.count_nonzero(eigenvalues > cutoff))
    condition_ratio = (
        float(max(0.0, eigenvalues[0]) / largest) if largest > 0.0 else 0.0
    )
    if not converged and reason is None:
        reason = "IRLS_NOT_CONVERGED"
    return LinearSeedResult(
        center=center,
        objective=float(previous),
        information=information,
        eigenvalues=eigenvalues,
        weak_direction=canonical_axis(eigenvectors[:, 0]),
        strong_direction=canonical_axis(eigenvectors[:, -1]),
        condition_ratio=condition_ratio,
        rank=rank,
        converged=converged,
        iterations=iterations,
        reason=reason,
    )


def common_center_seed(
    evidence: EvidenceView,
    *,
    loss: LossName,
    delta: float,
    max_iterations: int,
    step_tolerance: float,
) -> LinearSeedResult:
    """Convex radius-free common-centre initialisation."""

    evidence.validate()
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    squared_norm = np.sum(np.square(evidence.points), axis=1)
    for ring in evidence.active_rings:
        mask = evidence.ring_index == ring
        ring_weight = evidence.weight[mask]
        total = float(np.sum(ring_weight))
        if total <= 0.0:
            continue
        points = evidence.points[mask]
        mean_point = np.sum(ring_weight[:, None] * points, axis=0) / total
        mean_squared_norm = float(np.dot(ring_weight, squared_norm[mask]) / total)
        designs.append(2.0 * (points - mean_point))
        targets.append(squared_norm[mask] - mean_squared_norm)
        scales.append(evidence.sigma_alg[mask])
        masses.append(ring_weight)
    if not designs:
        raise ValueError("no usable parent ring for the convex seed")
    return solve_convex_irls(
        np.concatenate(designs, axis=0),
        np.concatenate(targets),
        np.concatenate(scales),
        np.concatenate(masses),
        loss=loss,
        delta=delta,
        max_iterations=max_iterations,
        step_tolerance=step_tolerance,
    )


def tangent_intersection_seed(
    evidence: EvidenceView,
    *,
    loss: LossName,
    delta: float,
    max_iterations: int,
    step_tolerance: float,
) -> LinearSeedResult:
    """Secondary seed using the correct tangent-zero projection equation."""

    evidence.validate()
    target = np.sum(evidence.tangents * evidence.points, axis=1)
    return solve_convex_irls(
        evidence.tangents,
        target,
        evidence.sigma_x,
        evidence.weight,
        loss=loss,
        delta=delta,
        max_iterations=max_iterations,
        step_tolerance=step_tolerance,
    )


class VarProProblem:
    """Robust common-centre objective with one exactly profiled radius per ring."""

    def __init__(
        self,
        evidence: EvidenceView,
        *,
        loss: LossName,
        delta: float,
        radius_tolerance: float,
        radius_max_iterations: int,
    ) -> None:
        evidence.validate()
        self.evidence = evidence
        self.loss = loss
        self.delta = float(delta)
        self.radius_tolerance = float(radius_tolerance)
        self.radius_max_iterations = int(radius_max_iterations)
        self._ring_masks = {
            int(ring): evidence.ring_index == ring for ring in evidence.active_rings
        }

    def robust_radius(self, distances: np.ndarray, mask: np.ndarray) -> float:
        return robust_location(
            distances[mask],
            self.evidence.sigma_x[mask],
            self.evidence.weight[mask],
            loss=self.loss,
            delta=self.delta,
            tolerance=self.radius_tolerance,
            max_iterations=self.radius_max_iterations,
        )

    def evaluate(self, center: np.ndarray | Sequence[float]) -> VarProEvaluation:
        point = _as_float64(center).reshape(2)
        if not np.all(np.isfinite(point)):
            return VarProEvaluation(
                objective=_HUGE_OBJECTIVE,
                gradient=np.zeros(2, dtype=np.float64),
                radii={},
            )
        offsets = point[None, :] - self.evidence.points
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        if not np.all(np.isfinite(distances)):
            return VarProEvaluation(
                objective=_HUGE_OBJECTIVE,
                gradient=np.zeros(2, dtype=np.float64),
                radii={},
            )
        objective = 0.0
        gradient = np.zeros(2, dtype=np.float64)
        radii: dict[int, float] = {}
        for ring, mask in self._ring_masks.items():
            radius = self.robust_radius(distances, mask)
            radii[ring] = radius
            standardised = (distances[mask] - radius) / self.evidence.sigma_x[mask]
            objective += float(
                np.sum(
                    self.evidence.weight[mask]
                    * loss_rho(standardised, self.loss, self.delta)
                )
            )
            radial_unit = np.zeros((int(np.count_nonzero(mask)), 2), dtype=np.float64)
            selected_distances = distances[mask]
            nonzero = selected_distances > 64.0 * _FLOAT_EPS
            radial_unit[nonzero] = offsets[mask][nonzero] / selected_distances[nonzero, None]
            gradient += np.sum(
                (
                    self.evidence.weight[mask]
                    * loss_psi(standardised, self.loss, self.delta)
                    / self.evidence.sigma_x[mask]
                )[:, None]
                * radial_unit,
                axis=0,
            )
        if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
            return VarProEvaluation(
                objective=_HUGE_OBJECTIVE,
                gradient=np.zeros(2, dtype=np.float64),
                radii={},
            )
        return VarProEvaluation(
            objective=float(objective), gradient=gradient, radii=radii
        )

    def objective(self, center: np.ndarray | Sequence[float]) -> float:
        return self.evaluate(center).objective

    def objective_gradient(
        self, center: np.ndarray | Sequence[float]
    ) -> tuple[float, np.ndarray]:
        evaluation = self.evaluate(center)
        return evaluation.objective, evaluation.gradient

    def far_objective(self, direction: np.ndarray | Sequence[float]) -> float:
        """Analytic per-ring profiled objective at infinity."""

        axis = canonical_axis(direction)
        projection = self.evidence.points @ axis
        objective = 0.0
        for mask in self._ring_masks.values():
            location = robust_location(
                projection[mask],
                self.evidence.sigma_x[mask],
                self.evidence.weight[mask],
                loss=self.loss,
                delta=self.delta,
                tolerance=self.radius_tolerance,
                max_iterations=self.radius_max_iterations,
            )
            standardised = (
                projection[mask] - location
            ) / self.evidence.sigma_x[mask]
            objective += float(
                np.sum(
                    self.evidence.weight[mask]
                    * loss_rho(standardised, self.loss, self.delta)
                )
            )
        return float(objective) if np.isfinite(objective) else _HUGE_OBJECTIVE

    def hessian(self, center: np.ndarray, relative_step: float = 2e-5) -> np.ndarray:
        """Symmetric finite-difference Hessian of the envelope gradient."""

        point = _as_float64(center).reshape(2)
        step = max(1e-7, float(relative_step) * (1.0 + float(np.linalg.norm(point))))
        hessian = np.zeros((2, 2), dtype=np.float64)
        for column in range(2):
            offset = np.zeros(2, dtype=np.float64)
            offset[column] = step
            plus = self.evaluate(point + offset).gradient
            minus = self.evaluate(point - offset).gradient
            hessian[:, column] = (plus - minus) / (2.0 * step)
        return 0.5 * (hessian + hessian.T)


def optimise_varpro(
    problem: VarProProblem,
    seed: np.ndarray | Sequence[float],
    *,
    max_iterations: int,
    gradient_tolerance: float,
    acceptable_gradient_tolerance: float,
    numerical_radius_limit: float,
) -> FiniteSolution:
    """Unbounded two-dimensional BFGS refinement from one seed."""

    initial = _as_float64(seed).reshape(2)
    if not np.all(np.isfinite(initial)):
        raise ValueError("VarPro seed must be finite")

    try:
        result = minimize(
            problem.objective_gradient,
            initial,
            method="BFGS",
            jac=True,
            options={
                "maxiter": int(max_iterations),
                "gtol": float(gradient_tolerance),
                "disp": False,
            },
        )
        center = _as_float64(result.x).reshape(2)
        evaluation = problem.evaluate(center)
        gradient_norm = float(np.linalg.norm(evaluation.gradient))
        escaped = bool(
            (not np.all(np.isfinite(center)))
            or float(np.linalg.norm(center)) > float(numerical_radius_limit)
            or evaluation.objective >= 0.5 * _HUGE_OBJECTIVE
        )
        converged = bool(
            not escaped
            and np.isfinite(evaluation.objective)
            and (bool(result.success) or gradient_norm <= acceptable_gradient_tolerance)
        )
        message = str(result.message)
        iterations = int(getattr(result, "nit", 0))
    except Exception as error:  # solver failures are returned as diagnostics
        center = initial.copy()
        evaluation = problem.evaluate(center)
        gradient_norm = float(np.linalg.norm(evaluation.gradient))
        escaped = False
        converged = False
        message = f"{type(error).__name__}: {error}"
        iterations = 0

    return FiniteSolution(
        center=center,
        objective=float(evaluation.objective),
        gradient_norm=gradient_norm,
        radii=evaluation.radii,
        converged=converged,
        escaped=escaped,
        iterations=iterations,
        seed=initial,
        message=message,
    )


def deduplicate_seeds(
    seeds: Iterable[np.ndarray | Sequence[float]], tolerance: float
) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for seed in seeds:
        point = _as_float64(seed).reshape(2)
        if not np.all(np.isfinite(point)):
            continue
        if all(float(np.linalg.norm(point - current)) > tolerance for current in unique):
            unique.append(point)
    return unique


def cluster_finite_solutions(
    solutions: Sequence[FiniteSolution], distance_tolerance: float
) -> list[FiniteSolution]:
    """Keep the lowest objective representative of each spatial basin."""

    ordered = sorted(
        (item for item in solutions if np.isfinite(item.objective) and not item.escaped),
        key=lambda item: item.objective,
    )
    representatives: list[FiniteSolution] = []
    for solution in ordered:
        if all(
            float(np.linalg.norm(solution.center - prior.center)) > distance_tolerance
            for prior in representatives
        ):
            representatives.append(solution)
    return representatives


def _local_minimum_indices(values: np.ndarray, circular: bool) -> list[int]:
    n = int(values.size)
    if n == 0:
        return []
    indices: list[int] = []
    for index in range(n):
        if not circular and (index == 0 or index == n - 1):
            continue
        left = values[(index - 1) % n]
        right = values[(index + 1) % n]
        if values[index] <= left and values[index] <= right:
            indices.append(index)
    if not indices:
        indices = [int(np.argmin(values))]
    return indices


def _select_separated_indices(
    values: np.ndarray, indices: Sequence[int], count: int, circular: bool
) -> list[int]:
    selected: list[int] = []
    n = int(values.size)
    for index in sorted(indices, key=lambda item: float(values[item])):
        if circular:
            separated = all(min(abs(index - old), n - abs(index - old)) > 1 for old in selected)
        else:
            separated = all(abs(index - old) > 1 for old in selected)
        if separated:
            selected.append(int(index))
        if len(selected) >= int(count):
            break
    return selected


def _scan_far_grid(
    problem: VarProProblem, angle_count: int, refine_lowest: int
) -> tuple[np.ndarray, np.ndarray, tuple[FarMinimum, ...]]:
    count = max(12, int(angle_count))
    angles = np.linspace(0.0, pi, count, endpoint=False, dtype=np.float64)
    objectives = np.array(
        [problem.far_objective((np.cos(phi), np.sin(phi))) for phi in angles],
        dtype=np.float64,
    )
    local = _local_minimum_indices(objectives, circular=True)
    chosen = _select_separated_indices(
        objectives, local, max(1, int(refine_lowest)), circular=True
    )
    step = pi / count
    minima: list[FarMinimum] = []
    for index in chosen:
        centre = float(angles[index])

        def wrapped(phi: float) -> float:
            wrapped_phi = float(phi % pi)
            return problem.far_objective((np.cos(wrapped_phi), np.sin(wrapped_phi)))

        refinement = minimize_scalar(
            wrapped,
            bounds=(centre - step, centre + step),
            method="bounded",
            options={"xatol": max(1e-10, step * 1e-7), "maxiter": 100},
        )
        if (
            bool(getattr(refinement, "success", False))
            and np.isfinite(getattr(refinement, "x", np.nan))
            and np.isfinite(getattr(refinement, "fun", np.nan))
        ):
            phi = float(refinement.x % pi)
            value = float(refinement.fun)
        else:
            phi = centre
            value = float(objectives[index])
        candidate = FarMinimum(phi_rad=phi, objective=value)
        if all(projective_angle_distance(phi, old.phi_rad) > 0.25 * step for old in minima):
            minima.append(candidate)
    if not minima:
        best_index = int(np.argmin(objectives))
        minima = [
            FarMinimum(
                phi_rad=float(angles[best_index]),
                objective=float(objectives[best_index]),
            )
        ]
    minima.sort(key=lambda item: item.objective)
    return angles, objectives, tuple(minima)


def _circular_cover_span(values: np.ndarray, period: float) -> float:
    wrapped = np.sort(np.unique(np.mod(_as_float64(values).reshape(-1), period)))
    if wrapped.size <= 1:
        return 0.0
    gaps = np.diff(np.concatenate((wrapped, np.array([wrapped[0] + period]))))
    return float(period - np.max(gaps))


def scan_far_field(
    problem: VarProProblem,
    *,
    angle_count: int,
    replay_factor: int,
    refine_lowest: int,
    support_delta: float,
    loss_equivalence: float,
) -> FarScanResult:
    angles, objectives, minima = _scan_far_grid(
        problem, angle_count, refine_lowest
    )
    replay_count = max(int(angle_count) + 1, int(angle_count) * max(2, int(replay_factor)))
    replay_angles, replay_objectives, replay_minima = _scan_far_grid(
        problem, replay_count, max(refine_lowest, 2 * refine_lowest)
    )
    best = min((*minima, *replay_minima), key=lambda item: item.objective)
    supported_candidates = [
        item
        for item in replay_minima
        if item.objective <= best.objective + float(support_delta)
    ]
    if not supported_candidates:
        supported_candidates = [replay_minima[0]]

    coarse_supported = [
        item for item in minima if item.objective <= minima[0].objective + support_delta
    ]
    coarse_step = pi / max(12, int(angle_count))
    direction_stable = all(
        any(
            projective_angle_distance(item.phi_rad, other.phi_rad)
            <= 2.0 * coarse_step
            for other in coarse_supported
        )
        for item in supported_candidates
    ) and all(
        any(
            projective_angle_distance(item.phi_rad, other.phi_rad)
            <= 2.0 * coarse_step
            for other in supported_candidates
        )
        for item in coarse_supported
    )
    loss_stable = abs(minima[0].objective - replay_minima[0].objective) <= max(
        float(loss_equivalence), 0.1 * float(support_delta)
    )
    coarse_supported_grid = angles[
        objectives <= best.objective + float(support_delta)
    ]
    replay_supported_grid = replay_angles[
        replay_objectives <= best.objective + float(support_delta)
    ]
    coarse_spread = _circular_cover_span(coarse_supported_grid, pi)
    support_axis_spread = _circular_cover_span(replay_supported_grid, pi)
    spread_stable = abs(coarse_spread - support_axis_spread) <= 2.0 * coarse_step
    stable = bool(direction_stable and loss_stable and spread_stable)
    reason = None if stable else "FAR_GRID_REPLAY_UNSTABLE"
    return FarScanResult(
        angles_rad=angles,
        objectives=objectives,
        replay_angles_rad=replay_angles,
        replay_objectives=replay_objectives,
        minima=minima,
        replay_minima=replay_minima,
        supported_minima=tuple(supported_candidates),
        best=best,
        stable=stable,
        angle_step_rad=coarse_step,
        support_axis_spread_rad=support_axis_spread,
        reason=reason,
    )


def _profile_value(
    problem: VarProProblem,
    anchor: np.ndarray,
    direction: np.ndarray,
    strong_direction: np.ndarray,
    z: float,
    strong_halfwidth: float,
) -> tuple[float, float, np.ndarray | None]:
    if abs(z) >= 1.0:
        signed_direction = direction if z > 0.0 else -direction
        return problem.far_objective(signed_direction), np.nan, None
    s = float(z) / (1.0 - abs(float(z)))
    base = anchor + s * direction

    def objective_eta(eta: float) -> float:
        return problem.objective(base + float(eta) * strong_direction)

    width = max(0.0, float(strong_halfwidth))
    if width <= 0.0:
        eta = 0.0
        value = objective_eta(eta)
    else:
        result = minimize_scalar(
            objective_eta,
            bounds=(-width, width),
            method="bounded",
            options={"xatol": max(1e-9, width * 1e-7), "maxiter": 100},
        )
        candidates = [
            (objective_eta(-width), -width),
            (objective_eta(0.0), 0.0),
            (objective_eta(width), width),
        ]
        if (
            bool(getattr(result, "success", False))
            and np.isfinite(getattr(result, "fun", np.nan))
            and np.isfinite(getattr(result, "x", np.nan))
        ):
            candidates.append((float(result.fun), float(result.x)))
        value, eta = min(candidates, key=lambda item: item[0])
    center = base + eta * strong_direction
    return float(value), float(eta), center


def _profile_components(
    z: np.ndarray, objectives: np.ndarray, threshold: float
) -> tuple[ProfileComponent, ...]:
    supported = objectives <= float(threshold)
    components: list[ProfileComponent] = []
    index = 0
    while index < supported.size:
        if not supported[index]:
            index += 1
            continue
        end = index
        while end + 1 < supported.size and supported[end + 1]:
            end += 1
        components.append(
            ProfileComponent(
                start_index=index,
                end_index=end,
                z_min=float(z[index]),
                z_max=float(z[end]),
                touches_negative_infinity=bool(index == 0 and z[index] <= -1.0),
                touches_positive_infinity=bool(
                    end == supported.size - 1 and z[end] >= 1.0
                ),
                objective_min=float(np.min(objectives[index : end + 1])),
            )
        )
        index = end + 1
    return tuple(components)


def _sample_direction_profile(
    problem: VarProProblem,
    anchor: np.ndarray,
    direction: np.ndarray,
    *,
    z_count: int,
    z_max: float,
    strong_halfwidth: float,
    refine_lowest: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = max(9, int(z_count))
    interior = np.linspace(-float(z_max), float(z_max), count, dtype=np.float64)
    z_values = np.concatenate((np.array([-1.0]), interior, np.array([1.0])))
    strong = canonical_axis(np.array([-direction[1], direction[0]]))

    sampled: list[tuple[float, float, float, np.ndarray | None]] = []
    for z_value in z_values:
        value, eta, center = _profile_value(
            problem,
            anchor,
            direction,
            strong,
            float(z_value),
            strong_halfwidth,
        )
        sampled.append((float(z_value), value, eta, center))

    sampled_objective = np.array([item[1] for item in sampled], dtype=np.float64)
    local = _local_minimum_indices(sampled_objective, circular=False)
    selected = _select_separated_indices(
        sampled_objective, local, max(1, int(refine_lowest)), circular=False
    )
    for index in selected:
        if index <= 0 or index >= len(sampled) - 1:
            continue
        lower = sampled[index - 1][0]
        upper = sampled[index + 1][0]
        if lower <= -1.0 or upper >= 1.0:
            continue

        def objective_z(z_value: float) -> float:
            return _profile_value(
                problem,
                anchor,
                direction,
                strong,
                float(z_value),
                strong_halfwidth,
            )[0]

        result = minimize_scalar(
            objective_z,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-8, "maxiter": 80},
        )
        if bool(result.success) and np.isfinite(result.fun):
            value, eta, center = _profile_value(
                problem,
                anchor,
                direction,
                strong,
                float(result.x),
                strong_halfwidth,
            )
            sampled.append((float(result.x), value, eta, center))

    sampled.sort(key=lambda item: item[0])
    z = np.array([item[0] for item in sampled], dtype=np.float64)
    objectives = np.array([item[1] for item in sampled], dtype=np.float64)
    eta = np.array([item[2] for item in sampled], dtype=np.float64)
    centers = np.full((len(sampled), 2), np.nan, dtype=np.float64)
    for index, item in enumerate(sampled):
        if item[3] is not None:
            centers[index] = item[3]
    return z, objectives, eta, centers


def profile_direction(
    problem: VarProProblem,
    anchor: np.ndarray,
    direction: np.ndarray,
    *,
    z_count: int,
    z_max: float,
    strong_halfwidth: float,
    refine_lowest: int,
    support_delta: float,
    reference_loss: float,
    loss_equivalence: float,
) -> DirectionProfileResult:
    axis = canonical_axis(direction)
    anchor64 = _as_float64(anchor).reshape(2)
    z, objectives, eta, centers = _sample_direction_profile(
        problem,
        anchor64,
        axis,
        z_count=z_count,
        z_max=z_max,
        strong_halfwidth=strong_halfwidth,
        refine_lowest=refine_lowest,
    )
    best = min(float(reference_loss), float(np.min(objectives)))
    threshold = best + float(support_delta)
    components = _profile_components(z, objectives, threshold)

    replay_count = max(int(z_count) + 2, 2 * int(z_count) - 1)
    replay_z, replay_objectives, _, _ = _sample_direction_profile(
        problem,
        anchor64,
        axis,
        z_count=replay_count,
        z_max=z_max,
        strong_halfwidth=strong_halfwidth,
        refine_lowest=max(refine_lowest, 2 * refine_lowest),
    )
    replay_best = min(float(reference_loss), float(np.min(replay_objectives)))
    replay_components = _profile_components(
        replay_z, replay_objectives, replay_best + float(support_delta)
    )
    endpoint_pattern = [
        (item.touches_negative_infinity, item.touches_positive_infinity)
        for item in components
    ]
    replay_endpoint_pattern = [
        (item.touches_negative_infinity, item.touches_positive_infinity)
        for item in replay_components
    ]
    stable = bool(
        len(components) == len(replay_components)
        and sorted(endpoint_pattern) == sorted(replay_endpoint_pattern)
        and abs(float(np.min(objectives)) - float(np.min(replay_objectives)))
        <= max(float(loss_equivalence), 0.1 * float(support_delta))
    )

    finite_seeds: list[np.ndarray] = []
    finite_mask = np.all(np.isfinite(centers), axis=1)
    local = _local_minimum_indices(objectives, circular=False)
    for index in sorted(local, key=lambda item: float(objectives[item])):
        if finite_mask[index] and objectives[index] <= threshold:
            candidate = centers[index]
            if all(float(np.linalg.norm(candidate - old)) > 1e-5 for old in finite_seeds):
                finite_seeds.append(candidate.copy())
    if not finite_seeds and np.any(finite_mask):
        finite_indices = np.flatnonzero(finite_mask)
        index = int(finite_indices[np.argmin(objectives[finite_indices])])
        finite_seeds.append(centers[index].copy())

    strong = canonical_axis(np.array([-axis[1], axis[0]]))
    supported = objectives <= threshold
    finite_eta = np.isfinite(eta)
    boundary_touched = bool(
        strong_halfwidth > 0.0
        and np.any(
            supported
            & finite_eta
            & (np.abs(eta) >= 0.98 * float(strong_halfwidth))
        )
    )
    return DirectionProfileResult(
        direction=axis,
        strong_direction=strong,
        anchor=anchor64,
        z=z,
        objectives=objectives,
        eta=eta,
        centers=centers,
        support_threshold=threshold,
        components=components,
        replay_component_count=len(replay_components),
        stable=stable,
        best_objective=float(np.min(objectives)),
        finite_seeds=tuple(finite_seeds),
        strong_halfwidth=float(strong_halfwidth),
        support_touches_strong_boundary=boundary_touched,
    )


def _compact_components(
    objectives: np.ndarray,
    directions: np.ndarray,
    radii: tuple[float | None, ...],
    threshold: float,
) -> tuple[CompactComponent, ...]:
    supported = objectives <= float(threshold)
    n_theta, n_radius = supported.shape
    visited = np.zeros_like(supported, dtype=bool)
    components: list[CompactComponent] = []

    for theta_index in range(n_theta):
        for radius_index in range(n_radius):
            if not supported[theta_index, radius_index] or visited[theta_index, radius_index]:
                continue
            stack = [(theta_index, radius_index)]
            visited[theta_index, radius_index] = True
            cells: list[tuple[int, int]] = []
            while stack:
                current_theta, current_radius = stack.pop()
                cells.append((current_theta, current_radius))
                neighbours = [
                    ((current_theta - 1) % n_theta, current_radius),
                    ((current_theta + 1) % n_theta, current_radius),
                ]
                if current_radius > 0:
                    neighbours.append((current_theta, current_radius - 1))
                if current_radius + 1 < n_radius:
                    neighbours.append((current_theta, current_radius + 1))
                # A circular common-centre objective has the same analytic
                # limit for +infinity*u and -infinity*u.  Those two boundary
                # cells are one unoriented axis, whose sign can only be
                # recovered later from trusted ring order.  The quotient is
                # applied at infinity only; finite antipodal centres remain
                # distinct points in R^2.
                if current_radius == n_radius - 1:
                    neighbours.append(
                        ((current_theta + n_theta // 2) % n_theta, current_radius)
                    )
                for neighbour_theta, neighbour_radius in neighbours:
                    if (
                        supported[neighbour_theta, neighbour_radius]
                        and not visited[neighbour_theta, neighbour_radius]
                    ):
                        visited[neighbour_theta, neighbour_radius] = True
                        stack.append((neighbour_theta, neighbour_radius))

            theta_indices = np.array([item[0] for item in cells], dtype=np.int64)
            radius_indices = np.array([item[1] for item in cells], dtype=np.int64)
            values = np.array(
                [objectives[t, r] for t, r in cells], dtype=np.float64
            )
            best_local = int(np.argmin(values))
            best_theta, best_radius_index = cells[best_local]
            best_radius = radii[best_radius_index]
            best_center = None
            if best_radius is not None:
                best_center = float(best_radius) * np.array(
                    [np.cos(directions[best_theta]), np.sin(directions[best_theta])],
                    dtype=np.float64,
                )
            finite_centers: list[np.ndarray] = []
            for cell_theta, cell_radius_index in cells:
                cell_radius = radii[cell_radius_index]
                if cell_radius is None:
                    continue
                finite_centers.append(
                    float(cell_radius)
                    * np.array(
                        [
                            np.cos(directions[cell_theta]),
                            np.sin(directions[cell_theta]),
                        ],
                        dtype=np.float64,
                    )
                )
            finite_span: float | None = None
            if finite_centers:
                finite_array = np.stack(finite_centers, axis=0)
                finite_span = float(
                    np.linalg.norm(
                        np.max(finite_array, axis=0) - np.min(finite_array, axis=0)
                    )
                )
            touches_infinity = bool(np.any(radius_indices == n_radius - 1))
            touches_radius_cap = bool(np.any(radius_indices == n_radius - 2))
            finite_radius_indices = sorted(
                int(index)
                for index in np.unique(radius_indices)
                if radii[int(index)] is not None
            )
            if finite_radius_indices:
                lower_index = max(0, finite_radius_indices[0] - 1)
                lower_value = radii[lower_index]
                component_radius_min = (
                    float(lower_value) if lower_value is not None else 0.0
                )
            else:
                component_radius_min = (
                    float(radii[-2])
                    if touches_infinity and radii[-2] is not None
                    else 0.0
                )
            component_radius_max: float | None
            if touches_infinity:
                component_radius_max = None
            elif finite_radius_indices:
                upper_index = min(n_radius - 2, finite_radius_indices[-1] + 1)
                upper_value = radii[upper_index]
                component_radius_max = (
                    float(upper_value) if upper_value is not None else None
                )
            else:
                component_radius_max = 0.0
            components.append(
                CompactComponent(
                    cell_count=len(cells),
                    touches_origin=bool(np.any(radius_indices == 0)),
                    touches_infinity=touches_infinity,
                    touches_radius_cap=touches_radius_cap,
                    radius_min=component_radius_min,
                    radius_max=component_radius_max,
                    theta_min_rad=float(np.min(directions[theta_indices])),
                    theta_max_rad=float(np.max(directions[theta_indices])),
                    objective_min=float(np.min(values)),
                    best_center=best_center,
                    best_theta_rad=float(directions[best_theta]),
                    finite_span_norm=finite_span,
                )
            )
    components.sort(key=lambda item: item.objective_min)
    return tuple(components)


def _sample_compact_grid(
    problem: VarProProblem,
    *,
    direction_count: int,
    radius_count: int,
    radius_min: float,
    radius_max: float,
) -> tuple[np.ndarray, tuple[float | None, ...], np.ndarray]:
    n_theta = max(24, int(direction_count))
    if n_theta % 2:
        n_theta += 1
    n_radius = max(8, int(radius_count))
    directions = np.linspace(0.0, 2.0 * pi, n_theta, endpoint=False, dtype=np.float64)
    finite_positive = np.geomspace(
        max(float(radius_min), 1e-12),
        max(float(radius_max), float(radius_min) * 2.0),
        n_radius - 2,
        dtype=np.float64,
    )
    radii: tuple[float | None, ...] = tuple(
        [0.0, *[float(value) for value in finite_positive], None]
    )
    objectives = np.empty((n_theta, len(radii)), dtype=np.float64)
    origin_value = problem.objective(np.zeros(2, dtype=np.float64))
    for theta_index, theta in enumerate(directions):
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        objectives[theta_index, 0] = origin_value
        for radius_index, radius in enumerate(radii[1:-1], start=1):
            objectives[theta_index, radius_index] = problem.objective(
                float(radius) * direction
            )
        objectives[theta_index, -1] = problem.far_objective(direction)
    return directions, radii, objectives


def compact_direction_distance_audit(
    problem: VarProProblem,
    *,
    direction_count: int,
    radius_count: int,
    radius_min: float,
    radius_max: float,
    replay_factor: int,
    support_delta: float,
    reference_loss: float,
    loss_equivalence: float,
) -> CompactAuditResult:
    directions, radii, objectives = _sample_compact_grid(
        problem,
        direction_count=direction_count,
        radius_count=radius_count,
        radius_min=radius_min,
        radius_max=radius_max,
    )
    best = min(float(reference_loss), float(np.min(objectives)))
    threshold = best + float(support_delta)
    components = _compact_components(objectives, directions, radii, threshold)

    factor = max(2, int(replay_factor))
    replay_directions, replay_radii, replay_objectives = _sample_compact_grid(
        problem,
        direction_count=int(direction_count) * factor,
        radius_count=int(radius_count) * factor,
        radius_min=radius_min,
        radius_max=radius_max,
    )
    replay_best = min(float(reference_loss), float(np.min(replay_objectives)))
    replay_components = _compact_components(
        replay_objectives,
        replay_directions,
        replay_radii,
        replay_best + float(support_delta),
    )
    endpoint_pattern = sorted(
        (item.touches_infinity, item.touches_radius_cap) for item in components
    )
    replay_endpoint_pattern = sorted(
        (item.touches_infinity, item.touches_radius_cap)
        for item in replay_components
    )
    stable = bool(
        len(components) == len(replay_components)
        and endpoint_pattern == replay_endpoint_pattern
        and abs(float(np.min(objectives)) - float(np.min(replay_objectives)))
        <= max(float(loss_equivalence), 0.1 * float(support_delta))
    )

    flat_index = int(np.argmin(objectives))
    theta_index, radius_index = np.unravel_index(flat_index, objectives.shape)
    best_radius = radii[radius_index]
    best_center = None
    if best_radius is not None:
        best_center = float(best_radius) * np.array(
            [np.cos(directions[theta_index]), np.sin(directions[theta_index])],
            dtype=np.float64,
        )
    return CompactAuditResult(
        directions_rad=directions,
        radii=radii,
        objectives=objectives,
        support_threshold=threshold,
        components=components,
        replay_component_count=len(replay_components),
        stable=stable,
        best_objective=float(np.min(objectives)),
        best_center=best_center,
    )


def local_support_geometry(
    problem: VarProProblem, center: np.ndarray, support_delta: float
) -> dict[str, Any]:
    """Quadratic local support summary; it never substitutes for far auditing."""

    hessian = problem.hessian(center)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = bool(np.all(eigenvalues > 0.0) and np.all(np.isfinite(eigenvalues)))
    if positive:
        semi_axes = np.sqrt(2.0 * float(support_delta) / eigenvalues)
    else:
        semi_axes = np.array([np.inf, np.inf], dtype=np.float64)
    return {
        "hessian": hessian,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "positive_definite": positive,
        "semi_axes": semi_axes,
        "major_span": float(np.max(semi_axes)),
        "minor_span": float(np.min(semi_axes)),
    }


def orient_axis_by_ring_order(
    problem: VarProProblem,
    axis: np.ndarray,
    *,
    reliable: bool,
    required_margin: float,
) -> tuple[np.ndarray | None, float, dict[str, Any]]:
    """Conditionally orient a far axis using trusted inner-to-outer ring order."""

    canonical = canonical_axis(axis)
    diagnostics: dict[str, Any] = {
        "ring_order_reliable": bool(reliable),
        "plus_consistency": None,
        "minus_consistency": None,
        "margin": 0.0,
    }
    if not reliable:
        return None, 0.0, diagnostics
    projection = problem.evidence.points @ canonical
    locations: list[tuple[int, float]] = []
    for ring in problem.evidence.active_rings:
        mask = problem.evidence.ring_index == ring
        location = robust_location(
            projection[mask],
            problem.evidence.sigma_x[mask],
            problem.evidence.weight[mask],
            loss=problem.loss,
            delta=problem.delta,
            tolerance=problem.radius_tolerance,
            max_iterations=problem.radius_max_iterations,
        )
        locations.append((int(ring), location))
    locations.sort(key=lambda item: item[0])
    if len(locations) < 2:
        return None, 0.0, diagnostics
    values = np.array([item[1] for item in locations], dtype=np.float64)
    differences = np.diff(values)
    denominator = float(np.sum(np.abs(differences)))
    if denominator <= 64.0 * _FLOAT_EPS:
        return None, 0.0, diagnostics
    # For c -> +infinity*axis, R_r = infinity - m_r(axis); hence outer
    # radii increase when successive projection locations decrease.
    plus = float(np.sum(np.maximum(-differences, 0.0)) / denominator)
    minus = float(np.sum(np.maximum(differences, 0.0)) / denominator)
    margin = abs(plus - minus)
    diagnostics.update(
        {
            "plus_consistency": plus,
            "minus_consistency": minus,
            "margin": margin,
            "projection_location_by_ring": [
                {"ring_index": ring, "location": float(value)}
                for ring, value in locations
            ],
        }
    )
    if margin <= max(float(required_margin), 32.0 * _FLOAT_EPS):
        return None, margin, diagnostics
    return (canonical if plus > minus else -canonical), margin, diagnostics
