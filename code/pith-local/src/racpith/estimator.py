"""GT-blind orchestration of the RAC-Pith v2 geometry estimator."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any, Mapping, Sequence

import numpy as np

from .config import FrozenConfig
from .contracts import EvidenceBundle, GeometryState, LocateResult
from .model_risk import assess_model_risk
from .numerics import (
    CompactAuditResult,
    DirectionProfileResult,
    EvidenceView,
    FarScanResult,
    FiniteSolution,
    LinearSeedResult,
    LossName,
    VarProProblem,
    canonical_axis,
    cluster_finite_solutions,
    common_center_seed,
    compact_direction_distance_audit,
    deduplicate_seeds,
    local_support_geometry,
    optimise_varpro,
    orient_axis_by_ring_order,
    profile_direction,
    projective_angle_distance,
    scan_far_field,
    tangent_intersection_seed,
)


def _json_finite(value: Any) -> Any:
    """Recursively convert arrays/scalars and replace non-finite floats by null."""

    if isinstance(value, np.ndarray):
        return _json_finite(value.tolist())
    if isinstance(value, np.generic):
        return _json_finite(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _native_xy(point: np.ndarray | Sequence[float] | None) -> list[float] | None:
    if point is None:
        return None
    array = np.asarray(point, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(array)):
        return None
    return [float(array[0]), float(array[1])]


def _axis_separation(a: np.ndarray, b: np.ndarray) -> float:
    phi_a = float(np.arctan2(a[1], a[0]) % pi)
    phi_b = float(np.arctan2(b[1], b[0]) % pi)
    return projective_angle_distance(phi_a, phi_b)


def _bounded_margin(value: float) -> float:
    """Keep an ordering diagnostic finite without changing its sign near zero."""

    number = float(value)
    if not np.isfinite(number):
        return -1.0e12
    return float(min(1.0e12, max(-1.0e12, number)))


@dataclass(frozen=True)
class _PreparedEvidence:
    view: EvidenceView
    active_mask: np.ndarray
    active_rings: tuple[int, ...]
    deleted_rings: tuple[int, ...]
    remaining_fraction: dict[int, float]


class RacPithEstimator:
    """Frozen, deterministic RAC-Pith v2 estimator.

    Ground truth is intentionally absent from both the constructor and
    :meth:`fit`.  Exact contribution code must call ``fit`` twice with different
    masks and only then compare the two results with GT in a separate module.
    """

    def __init__(self, config: FrozenConfig, *, run_id: str = "UNSPECIFIED") -> None:
        self.config = config
        self.run_id = str(run_id)
        self.solver = dict(config.section("solver"))
        self.search = dict(config.section("search"))
        self.state_config = dict(config.section("state"))
        loss = str(self.solver.get("loss", "pseudo_huber"))
        if loss not in {"pseudo_huber", "l2"}:
            raise ValueError("solver.loss must be 'pseudo_huber' or 'l2'")
        self.loss: LossName = loss  # type: ignore[assignment]
        self.delta = self._positive(self.solver, "huber_delta")
        self._validate_configuration()

    @staticmethod
    def _positive(section: Mapping[str, Any], key: str) -> float:
        value = float(section[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"configuration value {key!r} must be finite and positive")
        return value

    @staticmethod
    def _nonnegative(section: Mapping[str, Any], key: str) -> float:
        value = float(section[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"configuration value {key!r} must be finite and non-negative"
            )
        return value

    @staticmethod
    def _count(section: Mapping[str, Any], key: str, minimum: int) -> int:
        value = int(section[key])
        if value < minimum:
            raise ValueError(f"configuration value {key!r} must be >= {minimum}")
        return value

    def _validate_configuration(self) -> None:
        for key in (
            "irls_step_tol",
            "radius_tol",
            "outer_grad_tol",
            "acceptable_grad_tol",
            "numerical_radius_limit",
            "seed_merge_distance_norm",
            "mode_merge_distance_norm",
            "loss_equivalence",
        ):
            self._positive(self.solver, key)
        for key, minimum in (
            ("min_parent_rings", 1),
            ("irls_max_iter", 1),
            ("radius_max_iter", 1),
            ("outer_max_iter", 1),
        ):
            self._count(self.solver, key, minimum)
        fraction = float(self.solver["min_remaining_ring_fraction"])
        if not 0.0 < fraction <= 1.0:
            raise ValueError("min_remaining_ring_fraction must be in (0, 1]")
        for key in (
            "condition_ratio_trigger",
            "seed_disagreement_trigger_norm",
            "far_distance_trigger_norm",
            "far_gap_trigger",
            "far_gap_point_min",
            "support_delta",
            "profile_z_max",
            "profile_strong_halfwidth_norm",
            "profile_point_span_norm",
            "profile_range_span_norm",
            "compact_radius_min_norm",
            "compact_radius_max_norm",
        ):
            self._positive(self.search, key)
        if float(self.search["profile_z_max"]) >= 1.0:
            raise ValueError("profile_z_max must be strictly smaller than one")
        if float(self.search["compact_radius_max_norm"]) <= float(
            self.search["compact_radius_min_norm"]
        ):
            raise ValueError("compact radius maximum must exceed its minimum")
        if float(self.search["condition_ratio_trigger"]) > 1.0:
            raise ValueError("condition_ratio_trigger must not exceed one")
        for key, minimum in (
            ("far_angle_count", 12),
            ("far_refine_lowest", 1),
            ("far_grid_replay_factor", 2),
            ("profile_z_count", 9),
            ("profile_refine_lowest", 1),
            ("compact_direction_count", 24),
            ("compact_log_radius_count", 8),
            ("compact_replay_factor", 2),
            ("maximum_competing_directions", 1),
        ):
            self._count(self.search, key, minimum)
        axis_spread = self._nonnegative(
            self.state_config, "axis_direction_spread_deg"
        )
        if axis_spread > 90.0:
            raise ValueError("axis_direction_spread_deg must not exceed 90")
        ray_margin = self._nonnegative(self.state_config, "ray_order_margin")
        if ray_margin > 1.0:
            raise ValueError("ray_order_margin must be in [0, 1]")

    def _reject(
        self,
        evidence: EvidenceBundle,
        reason_codes: Sequence[str],
        diagnostics: Mapping[str, Any] | None = None,
    ) -> LocateResult:
        return LocateResult(
            crop_id=evidence.crop_id,
            state=GeometryState.REJECT,
            raw_center_norm=None,
            usable_center_norm=None,
            direction_norm=None,
            range_interval_norm=None,
            objective=None,
            radii_norm={},
            condition_ratio=None,
            weak_direction=None,
            finite_modes=[],
            far_scan={"status": "NOT_RUN"},
            profiles=[],
            search_adequate=False,
            model_risk=str(evidence.metadata.get("model_risk", "UNKNOWN")),
            production_usable=False,
            reason_codes=list(dict.fromkeys(str(item) for item in reason_codes)),
            diagnostics=_json_finite(dict(diagnostics or {})),
            config_hash=self.config.sha256,
            run_id=self.run_id,
        )

    def _prepare(
        self, evidence: EvidenceBundle, active_mask: np.ndarray | None
    ) -> _PreparedEvidence:
        evidence.validate()
        evidence_config_hash = evidence.metadata.get("evidence_config_hash")
        if (
            evidence_config_hash is not None
            and str(evidence_config_hash) != self.config.sha256
        ):
            raise ValueError(
                "evidence/config hash mismatch; rebuild candidate-independent evidence "
                "under the frozen configuration"
            )
        n = evidence.n_nodes
        if active_mask is None:
            mask = np.ones(n, dtype=bool)
        else:
            supplied = np.asarray(active_mask)
            if supplied.shape != (n,):
                raise ValueError(f"active_mask must have shape ({n},)")
            if supplied.dtype != np.bool_:
                if not np.all(np.isin(supplied, (0, 1))):
                    raise ValueError("active_mask must contain only booleans")
            mask = supplied.astype(bool, copy=True)
        if not np.any(mask):
            raise ValueError("active_mask deletes every evidence node")

        ring_count = len(evidence.ring_ids)
        full_mass = np.bincount(
            evidence.ring_index,
            weights=evidence.base_weight,
            minlength=ring_count,
        ).astype(np.float64)
        remaining_mass = np.bincount(
            evidence.ring_index[mask],
            weights=evidence.base_weight[mask],
            minlength=ring_count,
        ).astype(np.float64)
        fractions: dict[int, float] = {}
        active_rings: list[int] = []
        deleted_rings: list[int] = []
        minimum_fraction = float(self.solver["min_remaining_ring_fraction"])
        for ring in range(ring_count):
            if full_mass[ring] <= 0.0:
                continue
            fraction = float(remaining_mass[ring] / full_mass[ring])
            fractions[ring] = fraction
            if remaining_mass[ring] <= 32.0 * np.finfo(np.float64).eps * full_mass[ring]:
                deleted_rings.append(ring)
                continue
            if fraction + 1e-12 < minimum_fraction:
                raise ValueError(
                    f"ring {ring} retains fraction {fraction:.6g}, below the frozen minimum"
                )
            node_count = int(np.count_nonzero(mask & (evidence.ring_index == ring)))
            if node_count < 2:
                raise ValueError(f"ring {ring} has fewer than two active quadrature nodes")
            ring_points = np.asarray(
                evidence.points_norm[mask & (evidence.ring_index == ring)],
                dtype=np.float64,
            )
            spatial_extent = float(
                np.linalg.norm(np.max(ring_points, axis=0) - np.min(ring_points, axis=0))
            )
            if spatial_extent <= 1e-12:
                raise ValueError(f"ring {ring} has zero spatial extent")
            active_rings.append(ring)
        if len(active_rings) < int(self.solver["min_parent_rings"]):
            raise ValueError(
                "active evidence has fewer parent rings than the frozen minimum"
            )

        # Boolean slicing is the only deletion operation.  Crucially, base_weight
        # is copied verbatim: neither sibling subarcs nor surviving rings inherit
        # any deleted mass.
        view = EvidenceView(
            points=np.asarray(evidence.points_norm[mask], dtype=np.float64),
            tangents=np.asarray(evidence.tangents[mask], dtype=np.float64),
            sigma_x=np.asarray(evidence.sigma_x_norm[mask], dtype=np.float64),
            sigma_alg=np.asarray(evidence.sigma_alg_norm2[mask], dtype=np.float64),
            weight=np.asarray(evidence.base_weight[mask], dtype=np.float64),
            ring_index=np.asarray(evidence.ring_index[mask], dtype=np.int64),
        )
        view.validate()
        return _PreparedEvidence(
            view=view,
            active_mask=mask,
            active_rings=tuple(active_rings),
            deleted_rings=tuple(deleted_rings),
            remaining_fraction=fractions,
        )

    def objective_at_center(
        self,
        evidence: EvidenceBundle,
        center_norm: np.ndarray | Sequence[float],
        active_mask: np.ndarray | None = None,
    ) -> float:
        """Profile ring radii and score one fixed centre without refitting it.

        This is the authoritative post-hoc support-containment scorer used by
        evaluation.  It shares the exact loss, scales, radius solver and deletion
        semantics of :meth:`fit`, while never optimising the supplied centre.
        """

        center = np.asarray(center_norm, dtype=np.float64)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("center_norm must contain exactly two finite values")
        prepared = self._prepare(evidence, active_mask)
        problem = VarProProblem(
            prepared.view,
            loss=self.loss,
            delta=self.delta,
            radius_tolerance=float(self.solver["radius_tol"]),
            radius_max_iterations=int(self.solver["radius_max_iter"]),
        )
        value = float(problem.objective(center))
        if not np.isfinite(value):
            raise ValueError("profiled objective is non-finite at the supplied centre")
        return value

    def _optimise_seeds(
        self, problem: VarProProblem, seeds: Sequence[np.ndarray]
    ) -> list[FiniteSolution]:
        unique = deduplicate_seeds(
            seeds, float(self.solver["seed_merge_distance_norm"])
        )
        return [
            optimise_varpro(
                problem,
                seed,
                max_iterations=int(self.solver["outer_max_iter"]),
                gradient_tolerance=float(self.solver["outer_grad_tol"]),
                acceptable_gradient_tolerance=float(
                    self.solver["acceptable_grad_tol"]
                ),
                numerical_radius_limit=float(self.solver["numerical_radius_limit"]),
            )
            for seed in unique
        ]

    @staticmethod
    def _best_raw(
        problem: VarProProblem,
        solutions: Sequence[FiniteSolution],
        fallback_center: np.ndarray,
    ) -> FiniteSolution:
        candidates = [
            item
            for item in solutions
            if not item.escaped
            and np.all(np.isfinite(item.center))
            and np.isfinite(item.objective)
        ]
        converged = [item for item in candidates if item.converged]
        if converged:
            return min(converged, key=lambda item: item.objective)
        if candidates:
            return min(candidates, key=lambda item: item.objective)
        evaluation = problem.evaluate(fallback_center)
        return FiniteSolution(
            center=fallback_center.copy(),
            objective=float(evaluation.objective),
            gradient_norm=float(np.linalg.norm(evaluation.gradient)),
            radii=evaluation.radii,
            converged=False,
            escaped=False,
            iterations=0,
            seed=fallback_center.copy(),
            message="SEED_EVALUATION_ONLY",
        )

    @staticmethod
    def _profile_directions(
        weak_direction: np.ndarray,
        far_scan: FarScanResult,
        *,
        maximum: int,
        separation_rad: float,
    ) -> tuple[list[np.ndarray], bool]:
        directions = [canonical_axis(weak_direction)]
        overflow = False
        for minimum in far_scan.supported_minima:
            candidate = minimum.direction
            if all(_axis_separation(candidate, old) > separation_rad for old in directions):
                if len(directions) >= maximum:
                    overflow = True
                    continue
                directions.append(candidate)
        return directions, overflow

    def _run_profiles(
        self,
        problem: VarProProblem,
        anchor: np.ndarray,
        directions: Sequence[np.ndarray],
        reference_loss: float,
    ) -> list[DirectionProfileResult]:
        maximum_halfwidth = float(self.search["profile_strong_halfwidth_norm"])
        support_delta = float(self.search["support_delta"])
        try:
            local_hessian = problem.hessian(anchor)
        except Exception:
            local_hessian = np.full((2, 2), np.nan, dtype=np.float64)
        output: list[DirectionProfileResult] = []
        for direction in directions:
            axis = canonical_axis(direction)
            strong = np.array([-axis[1], axis[0]], dtype=np.float64)
            curvature = float(strong @ local_hessian @ strong)
            if np.isfinite(curvature) and curvature > 0.0:
                local_halfwidth = float(np.sqrt(2.0 * support_delta / curvature))
                strong_halfwidth = min(
                    maximum_halfwidth,
                    max(1e-5, 1.25 * local_halfwidth),
                )
            else:
                strong_halfwidth = maximum_halfwidth
            output.append(
                profile_direction(
                    problem,
                    anchor,
                    axis,
                    z_count=int(self.search["profile_z_count"]),
                    z_max=float(self.search["profile_z_max"]),
                    strong_halfwidth=strong_halfwidth,
                    refine_lowest=int(self.search["profile_refine_lowest"]),
                    support_delta=support_delta,
                    reference_loss=float(reference_loss),
                    loss_equivalence=float(self.solver["loss_equivalence"]),
                )
            )
        return output

    def _run_compact(
        self, problem: VarProProblem, reference_loss: float
    ) -> CompactAuditResult:
        return compact_direction_distance_audit(
            problem,
            direction_count=int(self.search["compact_direction_count"]),
            radius_count=int(self.search["compact_log_radius_count"]),
            radius_min=float(self.search["compact_radius_min_norm"]),
            radius_max=float(self.search["compact_radius_max_norm"]),
            replay_factor=int(self.search["compact_replay_factor"]),
            support_delta=float(self.search["support_delta"]),
            reference_loss=float(reference_loss),
            loss_equivalence=float(self.solver["loss_equivalence"]),
        )

    @staticmethod
    def _range_from_profiles(
        profiles: Sequence[DirectionProfileResult], threshold: float
    ) -> tuple[float, float | None] | None:
        finite_radius: list[float] = []
        touches_infinity = False
        for profile in profiles:
            supported = profile.objectives <= float(threshold)
            for index in np.flatnonzero(supported):
                center = profile.centers[index]
                if np.all(np.isfinite(center)):
                    finite_radius.append(float(np.linalg.norm(center)))
                elif profile.z[index] <= -1.0 or profile.z[index] >= 1.0:
                    touches_infinity = True
        if not finite_radius:
            return (0.0, None) if touches_infinity else None
        return min(finite_radius), None if touches_infinity else max(finite_radius)

    @staticmethod
    def _profile_finite_span(
        profiles: Sequence[DirectionProfileResult], threshold: float
    ) -> float:
        supported_centers: list[np.ndarray] = []
        for profile in profiles:
            supported = profile.objectives <= float(threshold)
            for index in np.flatnonzero(supported):
                center = profile.centers[index]
                if np.all(np.isfinite(center)):
                    supported_centers.append(center)
        if len(supported_centers) < 2:
            return 0.0
        values = np.stack(supported_centers, axis=0)
        return float(np.linalg.norm(np.max(values, axis=0) - np.min(values, axis=0)))

    def _mode_dict(
        self, solution: FiniteSolution, evidence: EvidenceBundle
    ) -> dict[str, Any]:
        payload = solution.as_dict()
        payload.pop("radii_norm_by_index", None)
        payload["radii_norm"] = {
            evidence.ring_ids[index]: float(radius)
            for index, radius in solution.radii.items()
            if 0 <= index < len(evidence.ring_ids)
        }
        return _json_finite(payload)

    def fit(
        self,
        evidence: EvidenceBundle,
        active_mask: np.ndarray | None = None,
    ) -> LocateResult:
        """Fit one crop, optionally after an exact evidence deletion.

        ``active_mask`` is a row-selection mask only.  Retained quadrature
        weights are the original values from the ``EvidenceBundle`` (converted
        to float64 only), without any rescaling.
        The method accepts no GT coordinate.
        """

        reason_codes: list[str] = []
        try:
            prepared = self._prepare(evidence, active_mask)
        except Exception as error:
            return self._reject(
                evidence,
                ["INVALID_OR_INSUFFICIENT_EVIDENCE"],
                {"prepare_error": f"{type(error).__name__}: {error}"},
            )

        view = prepared.view
        try:
            common_seed = common_center_seed(
                view,
                loss=self.loss,
                delta=self.delta,
                max_iterations=int(self.solver["irls_max_iter"]),
                step_tolerance=float(self.solver["irls_step_tol"]),
            )
        except Exception as error:
            return self._reject(
                evidence,
                ["CONVEX_INITIALISATION_FAILED"],
                {"initialisation_error": f"{type(error).__name__}: {error}"},
            )
        if common_seed.rank == 0:
            return self._reject(
                evidence,
                ["NO_DIRECTIONAL_INFORMATION"],
                {"common_seed": self._seed_diagnostics(common_seed)},
            )
        if not common_seed.converged:
            reason_codes.append(common_seed.reason or "COMMON_IRLS_NOT_CONVERGED")

        tangent_seed: LinearSeedResult | None = None
        tangent_error: str | None = None
        try:
            tangent_seed = tangent_intersection_seed(
                view,
                loss=self.loss,
                delta=self.delta,
                max_iterations=int(self.solver["irls_max_iter"]),
                step_tolerance=float(self.solver["irls_step_tol"]),
            )
            if not tangent_seed.converged:
                reason_codes.append(tangent_seed.reason or "TANGENT_IRLS_NOT_CONVERGED")
        except Exception as error:
            tangent_error = f"{type(error).__name__}: {error}"
            reason_codes.append("TANGENT_SEED_UNAVAILABLE")

        problem = VarProProblem(
            view,
            loss=self.loss,
            delta=self.delta,
            radius_tolerance=float(self.solver["radius_tol"]),
            radius_max_iterations=int(self.solver["radius_max_iter"]),
        )
        seeds: list[np.ndarray] = [
            common_seed.center,
            np.zeros(2, dtype=np.float64),
        ]
        if tangent_seed is not None:
            seeds.extend(
                [
                    tangent_seed.center,
                    0.5 * (common_seed.center + tangent_seed.center),
                ]
            )
        # Symmetric weak-axis starts add no prior: they only expose distinct
        # finite basins and are rescored by exactly the same profiled objective.
        seed_offset = max(
            float(self.search["seed_disagreement_trigger_norm"]),
            0.05 * (1.0 + float(np.linalg.norm(common_seed.center))),
        )
        seeds.extend(
            [
                common_seed.center + seed_offset * common_seed.weak_direction,
                common_seed.center - seed_offset * common_seed.weak_direction,
            ]
        )
        finite_trials = self._optimise_seeds(problem, seeds)
        accepted = [item for item in finite_trials if item.converged and not item.escaped]
        modes = cluster_finite_solutions(
            accepted, float(self.solver["mode_merge_distance_norm"])
        )
        raw_best = self._best_raw(problem, finite_trials, common_seed.center)

        try:
            far_scan = scan_far_field(
                problem,
                angle_count=int(self.search["far_angle_count"]),
                replay_factor=int(self.search["far_grid_replay_factor"]),
                refine_lowest=int(self.search["far_refine_lowest"]),
                support_delta=float(self.search["support_delta"]),
                loss_equivalence=float(self.solver["loss_equivalence"]),
            )
        except Exception as error:
            return self._reject(
                evidence,
                [*reason_codes, "FAR_FIELD_SCAN_FAILED"],
                {
                    "far_scan_error": f"{type(error).__name__}: {error}",
                    "common_seed": self._seed_diagnostics(common_seed),
                },
            )
        if not far_scan.stable:
            reason_codes.append(far_scan.reason or "FAR_GRID_REPLAY_UNSTABLE")

        seed_disagreement = (
            float(np.linalg.norm(common_seed.center - tangent_seed.center))
            if tangent_seed is not None
            else None
        )
        far_gap = float(far_scan.best.objective - raw_best.objective)
        trigger_reasons: list[str] = []
        if common_seed.rank < 2 or common_seed.condition_ratio < float(
            self.search["condition_ratio_trigger"]
        ):
            trigger_reasons.append("WEAK_CONVEX_DIRECTION")
        if seed_disagreement is not None and seed_disagreement > float(
            self.search["seed_disagreement_trigger_norm"]
        ):
            trigger_reasons.append("SEED_DISAGREEMENT")
        if float(np.linalg.norm(raw_best.center)) > float(
            self.search["far_distance_trigger_norm"]
        ):
            trigger_reasons.append("FINITE_SOLUTION_FAR_FROM_CROP")
        if far_gap < float(self.search["far_gap_trigger"]):
            trigger_reasons.append("FINITE_FAR_GAP_SMALL")
        if not far_scan.stable:
            trigger_reasons.append("FAR_REPLAY_REQUIRES_PROFILE")
        if not accepted:
            trigger_reasons.append("FINITE_OPTIMISER_UNRELIABLE")

        local_support: dict[str, Any] | None = None
        try:
            local_support = local_support_geometry(
                problem, raw_best.center, float(self.search["support_delta"])
            )
            if not bool(local_support["positive_definite"]):
                trigger_reasons.append("FINITE_HESSIAN_NOT_POSITIVE")
            elif float(local_support["major_span"]) > float(
                self.search["profile_point_span_norm"]
            ):
                trigger_reasons.append("FINITE_SUPPORT_WIDE")
        except Exception as error:
            trigger_reasons.append("FINITE_HESSIAN_FAILED")
            reason_codes.append("FINITE_HESSIAN_FAILED")
            local_support = {"error": f"{type(error).__name__}: {error}"}

        axis_separation = max(
            2.0 * far_scan.angle_step_rad,
            np.deg2rad(float(self.state_config["axis_direction_spread_deg"])),
        )
        profile_directions, direction_overflow = self._profile_directions(
            common_seed.weak_direction,
            far_scan,
            maximum=int(self.search["maximum_competing_directions"]),
            separation_rad=float(axis_separation),
        )
        if direction_overflow:
            trigger_reasons.append("TOO_MANY_COMPETING_FAR_DIRECTIONS")

        profiles: list[DirectionProfileResult] = []
        if trigger_reasons:
            reference_loss = min(raw_best.objective, far_scan.best.objective)
            try:
                profiles = self._run_profiles(
                    problem, raw_best.center, profile_directions, reference_loss
                )
            except Exception as error:
                return self._reject(
                    evidence,
                    [*reason_codes, "DISTANCE_PROFILE_FAILED"],
                    {
                        "profile_error": f"{type(error).__name__}: {error}",
                        "trigger_reasons": trigger_reasons,
                        "far_scan": far_scan.as_dict(),
                    },
                )

            profile_seeds = [seed for profile in profiles for seed in profile.finite_seeds]
            if profile_seeds:
                new_trials = self._optimise_seeds(problem, profile_seeds)
                finite_trials.extend(new_trials)
                accepted = [
                    item for item in finite_trials if item.converged and not item.escaped
                ]
                modes = cluster_finite_solutions(
                    accepted, float(self.solver["mode_merge_distance_norm"])
                )
                new_raw = self._best_raw(problem, finite_trials, common_seed.center)
                center_changed = float(
                    np.linalg.norm(new_raw.center - raw_best.center)
                ) > float(self.solver["seed_merge_distance_norm"])
                objective_changed = new_raw.objective < raw_best.objective - (
                    1e-12 * max(1.0, abs(raw_best.objective))
                )
                if center_changed or objective_changed:
                    raw_best = new_raw
                    # The strong-direction interval is anchored at the final
                    # finite candidate, so a material shift requires one replay.
                    profiles = self._run_profiles(
                        problem,
                        raw_best.center,
                        profile_directions,
                        min(raw_best.objective, far_scan.best.objective),
                    )
                else:
                    raw_best = new_raw
            if not all(profile.stable for profile in profiles):
                reason_codes.append("PROFILE_REPLAY_UNSTABLE")

        support_delta = float(self.search["support_delta"])
        global_reference = min(raw_best.objective, far_scan.best.objective)
        supported_modes = [
            item for item in modes if item.objective <= global_reference + support_delta
        ]
        far_supported = far_scan.best.objective <= raw_best.objective + support_delta
        profile_multicomponent = any(len(profile.components) > 1 for profile in profiles)
        profile_unstable = any(not profile.stable for profile in profiles)
        profile_strong_boundary = any(
            profile.support_touches_strong_boundary for profile in profiles
        )
        profile_span = self._profile_finite_span(
            profiles, global_reference + support_delta
        )
        profile_exceeds_compact_range = bool(
            profiles
            and profile_span > float(self.search["profile_range_span_norm"])
        )
        local_bad = bool(
            local_support is None
            or not bool(local_support.get("positive_definite", False))
        )
        need_compact = bool(
            direction_overflow
            or not far_scan.stable
            or profile_unstable
            or profile_strong_boundary
            or profile_multicomponent
            or profile_exceeds_compact_range
            or len(supported_modes) > 1
            or far_supported
            or (bool(trigger_reasons) and local_bad)
            or not accepted
        )

        compact: CompactAuditResult | None = None
        if need_compact:
            try:
                compact = self._run_compact(problem, global_reference)
                if compact.best_center is not None:
                    compact_trials = self._optimise_seeds(
                        problem, [compact.best_center]
                    )
                    old_reference = global_reference
                    finite_trials.extend(compact_trials)
                    accepted = [
                        item
                        for item in finite_trials
                        if item.converged and not item.escaped
                    ]
                    modes = cluster_finite_solutions(
                        accepted, float(self.solver["mode_merge_distance_norm"])
                    )
                    raw_best = self._best_raw(problem, finite_trials, common_seed.center)
                    global_reference = min(raw_best.objective, far_scan.best.objective)
                    reference_tolerance = 1e-12 * max(1.0, abs(old_reference))
                    if global_reference < old_reference - reference_tolerance:
                        compact = self._run_compact(problem, global_reference)
                if not compact.stable:
                    reason_codes.append("COMPACT_AUDIT_REPLAY_UNSTABLE")
            except Exception as error:
                return self._reject(
                    evidence,
                    [*reason_codes, "COMPACT_AUDIT_FAILED"],
                    {
                        "compact_error": f"{type(error).__name__}: {error}",
                        "trigger_reasons": trigger_reasons,
                        "far_scan": far_scan.as_dict(),
                        "profiles": [profile.as_dict() for profile in profiles],
                    },
                )

        # Refresh quantities after all profile/audit seeds have been refined.
        raw_best = self._best_raw(problem, finite_trials, common_seed.center)
        accepted = [item for item in finite_trials if item.converged and not item.escaped]
        modes = cluster_finite_solutions(
            accepted, float(self.solver["mode_merge_distance_norm"])
        )
        global_reference = min(raw_best.objective, far_scan.best.objective)
        supported_modes = [
            item for item in modes if item.objective <= global_reference + support_delta
        ]
        profile_span = self._profile_finite_span(
            profiles, global_reference + support_delta
        )
        far_gap = float(far_scan.best.objective - raw_best.objective)
        try:
            local_support = local_support_geometry(
                problem, raw_best.center, support_delta
            )
        except Exception as error:
            local_support = {"error": f"{type(error).__name__}: {error}"}

        compact_resolves = bool(compact is not None and compact.stable)
        profiles_resolve = bool(all(profile.stable for profile in profiles))
        compact_cap_truncated = bool(
            compact is not None
            and any(
                component.touches_radius_cap and not component.touches_infinity
                for component in compact.components
            )
        )
        search_adequate = bool(
            (far_scan.stable or compact_resolves)
            and (profiles_resolve or compact_resolves)
            and (compact is None or compact.stable)
            and not compact_cap_truncated
        )
        point_span_limit = float(self.search["profile_point_span_norm"])
        far_gap_limit = float(self.search["far_gap_point_min"])
        local_major_span = float(local_support.get("major_span", np.inf))
        local_margin = (
            (point_span_limit - local_major_span) / max(point_span_limit, 1e-15)
            if bool(local_support.get("positive_definite", False))
            and np.isfinite(local_major_span)
            else -1.0
        )
        far_margin = _bounded_margin(
            (far_gap - far_gap_limit) / max(far_gap_limit, 1e-15)
        )
        profile_margin: float | None = None
        if profiles:
            profile_topology_ok = bool(
                all(len(profile.components) <= 1 for profile in profiles)
                and all(
                    not any(
                        component.touches_negative_infinity
                        or component.touches_positive_infinity
                        for component in profile.components
                    )
                    for profile in profiles
                )
            )
            profile_margin = _bounded_margin(
                (2.0 * point_span_limit - profile_span)
                / max(2.0 * point_span_limit, 1e-15)
                if profile_topology_ok
                else -1.0
            )
        compact_margin: float | None = None
        if compact is not None:
            compact_topology_ok = bool(
                len(compact.components) == 1
                and not compact.components[0].touches_infinity
                and not compact_cap_truncated
            )
            compact_span_for_margin = (
                compact.components[0].finite_span_norm
                if compact_topology_ok
                else None
            )
            compact_margin = _bounded_margin(
                (2.0 * point_span_limit - float(compact_span_for_margin))
                / max(2.0 * point_span_limit, 1e-15)
                if compact_span_for_margin is not None
                else -1.0
            )
        point_safety_components = {
            "far_gap": _bounded_margin(far_margin),
            "local_tightness": _bounded_margin(local_margin),
            "profile_tightness": profile_margin,
            "compact_tightness": compact_margin,
            "search_replay": 1.0 if search_adequate else -1.0,
            "finite_solver": 1.0 if accepted else -1.0,
        }
        point_safety_margin = float(
            min(
                value
                for value in point_safety_components.values()
                if value is not None
            )
        )

        state = GeometryState.REJECT
        direction: np.ndarray | None = None
        distance_range: tuple[float, float | None] | None = None
        orientation_diagnostics: dict[str, Any] = {}
        point_gate: dict[str, Any] = {}

        if not search_adequate:
            if compact_cap_truncated:
                reason_codes.append("COMPACT_RADIUS_CAP_TOUCHED")
            reason_codes.append("SEARCH_INADEQUATE")
        else:
            authoritative_components = compact.components if compact is not None else ()
            if compact is not None and len(authoritative_components) > 1:
                state = GeometryState.MULTIMODAL
                reason_codes.append("MULTIPLE_SUPPORTED_COMPONENTS")
            elif compact is not None and len(authoritative_components) == 0:
                state = GeometryState.REJECT
                search_adequate = False
                reason_codes.append("EMPTY_COMPACT_SUPPORT")
            elif compact is not None and authoritative_components[0].touches_infinity:
                axis = far_scan.best.direction
                maximum_axis_spread = np.deg2rad(
                    float(self.state_config["axis_direction_spread_deg"])
                )
                if far_scan.support_axis_spread_rad > maximum_axis_spread:
                    state = GeometryState.REJECT
                    search_adequate = False
                    reason_codes.append("AXIS_DIRECTION_UNSTABLE")
                else:
                    oriented, _, orientation_diagnostics = orient_axis_by_ring_order(
                        problem,
                        axis,
                        reliable=evidence.metadata.get("ring_order_reliable") is True,
                        required_margin=float(self.state_config["ray_order_margin"]),
                    )
                    component = authoritative_components[0]
                    distance_range = (float(component.radius_min), None)
                    if oriented is not None:
                        state = GeometryState.RAY
                        direction = oriented
                        reason_codes.append("FAR_SUPPORT_ORIENTED_BY_RING_ORDER")
                    else:
                        state = GeometryState.AXIS
                        direction = canonical_axis(axis)
                        reason_codes.append("FAR_SUPPORT_UNORIENTED")
            elif compact is None and far_scan.best.objective <= raw_best.objective + support_delta:
                # This should normally have triggered compact auditing.  Refuse
                # to infer connectivity between a finite and a far component.
                state = GeometryState.REJECT
                search_adequate = False
                reason_codes.append("FINITE_FAR_CONNECTIVITY_UNAUDITED")
            elif len(supported_modes) > 1:
                state = GeometryState.MULTIMODAL
                reason_codes.append("MULTIPLE_FINITE_MODES")
            elif not accepted:
                state = GeometryState.REJECT
                search_adequate = False
                reason_codes.append("FINITE_SOLVER_UNSTABLE")
            else:
                positive_hessian = bool(local_support.get("positive_definite", False))
                major_span = float(local_support.get("major_span", np.inf))
                finite_closed = bool(
                    compact is None
                    or (
                        len(compact.components) == 1
                        and not compact.components[0].touches_infinity
                    )
                )
                compact_span = (
                    compact.components[0].finite_span_norm
                    if compact is not None and len(compact.components) == 1
                    else None
                )
                compact_tight = bool(
                    compact_span is None
                    or compact_span
                    <= 2.0 * float(self.search["profile_point_span_norm"])
                )
                profile_tight = bool(
                    not profiles
                    or profile_span
                    <= 2.0 * float(self.search["profile_point_span_norm"])
                )
                point_gate = {
                    "finite_solver_converged": True,
                    "local_hessian_positive": positive_hessian,
                    "local_major_span_norm": major_span,
                    "point_span_limit_norm": float(
                        self.search["profile_point_span_norm"]
                    ),
                    "far_gap": far_gap,
                    "far_gap_point_min": float(self.search["far_gap_point_min"]),
                    "far_replay_stable": bool(far_scan.stable or compact_resolves),
                    "finite_support_closed": finite_closed,
                    "compact_finite_span_norm": compact_span,
                    "compact_support_tight": compact_tight,
                    "profile_finite_span_norm": profile_span,
                    "profile_support_tight": profile_tight,
                    "point_safety_margin": point_safety_margin,
                }
                point_ok = bool(
                    positive_hessian
                    and major_span <= float(self.search["profile_point_span_norm"])
                    and far_gap >= float(self.search["far_gap_point_min"])
                    and (far_scan.stable or compact_resolves)
                    and finite_closed
                    and compact_tight
                    and profile_tight
                )
                if point_ok:
                    state = GeometryState.POINT
                else:
                    state = GeometryState.RANGE
                    center_radius = float(np.linalg.norm(raw_best.center))
                    if center_radius > max(1e-14, major_span):
                        direction = raw_best.center / center_radius
                    else:
                        direction = None
                    if compact is not None and compact.components:
                        component = compact.components[0]
                        distance_range = (
                            float(component.radius_min),
                            (
                                float(component.radius_max)
                                if component.radius_max is not None
                                else None
                            ),
                        )
                    else:
                        profiled_range = self._range_from_profiles(
                            profiles, global_reference + support_delta
                        )
                        if profiled_range is not None:
                            distance_range = profiled_range
                        else:
                            radius = center_radius
                            distance_range = (
                                max(0.0, radius - major_span),
                                radius + major_span if np.isfinite(major_span) else None,
                            )
                    reason_codes.append("FINITE_SUPPORT_NOT_POINT_TIGHT")

        point_safety_components["search_replay"] = (
            1.0 if search_adequate else -1.0
        )
        point_safety_margin = float(
            min(
                value
                for value in point_safety_components.values()
                if value is not None
            )
        )
        if point_gate:
            point_gate["point_safety_margin"] = point_safety_margin

        model_risk_diagnostics = assess_model_risk(
            view,
            np.asarray(raw_best.center, dtype=np.float64),
            raw_best.radii,
            self.config.section("model_risk"),
        )
        model_risk = str(model_risk_diagnostics["level"])
        target_domain = str(
            evidence.metadata.get("target_domain", "TARGET_UNKNOWN")
        )
        calibration_status = str(
            self.config.data.get("calibration_status", "UNKNOWN")
        )
        calibration_frozen = bool(
            "FROZEN" in calibration_status.upper()
            and "PROVISIONAL" not in calibration_status.upper()
        )
        production_usable = bool(
            state == GeometryState.POINT
            and search_adequate
            and model_risk == "LOW_RISK"
            and target_domain == "TARGET_ALIGNED"
            and calibration_frozen
        )
        if state == GeometryState.POINT and not production_usable:
            if not calibration_frozen:
                reason_codes.append("CONFIG_NOT_DEVELOPMENT_FROZEN")
            if model_risk != "LOW_RISK":
                reason_codes.append("MODEL_RISK_NOT_CLEARED")
            if target_domain != "TARGET_ALIGNED":
                reason_codes.append("TARGET_DOMAIN_NOT_ALIGNED")

        usable_center = raw_best.center if state == GeometryState.POINT else None
        radii = {
            evidence.ring_ids[index]: float(radius)
            for index, radius in raw_best.radii.items()
            if 0 <= index < len(evidence.ring_ids) and np.isfinite(radius)
        }
        active_mass = {
            evidence.ring_ids[index]: float(
                np.sum(view.weight[view.ring_index == index])
            )
            for index in prepared.active_rings
        }
        diagnostics = {
            "numeric_dtype": "float64",
            "loss": self.loss,
            "huber_delta": self.delta,
            "common_seed": self._seed_diagnostics(common_seed),
            "tangent_seed": (
                self._seed_diagnostics(tangent_seed)
                if tangent_seed is not None
                else None
            ),
            "tangent_seed_error": tangent_error,
            "seed_disagreement_norm": seed_disagreement,
            "finite_trials": [item.as_dict() for item in finite_trials],
            "finite_converged_count": len(accepted),
            "far_gap": far_gap,
            "global_support_reference": global_reference,
            "support_delta": support_delta,
            "profile_triggered": bool(trigger_reasons),
            "profile_trigger_reasons": trigger_reasons,
            "compact_audit_required": need_compact,
            "profile_finite_span_norm": profile_span,
            "profile_exceeds_compact_range": profile_exceeds_compact_range,
            "profile_support_touches_strong_boundary": profile_strong_boundary,
            "compact_radius_cap_truncated": compact_cap_truncated,
            "compact_audit": compact.as_dict() if compact is not None else None,
            "local_support": local_support,
            "point_gate": point_gate,
            "state_decision_trace": {
                "mutually_exclusive_priority": [
                    "REJECT_INVALID_NUMERICAL_OR_SEARCH",
                    "MULTIMODAL_MULTIPLE_SUPPORT_COMPONENTS",
                    "AXIS_OR_RAY_INFINITY_SUPPORT",
                    "RANGE_FINITE_BUT_WIDE",
                    "POINT_FINITE_TIGHT_AND_STABLE",
                ],
                "selected_state": state.value,
                "search_adequate": search_adequate,
            },
            "point_safety_margin": point_safety_margin,
            "point_safety_margin_components": point_safety_components,
            "point_safety_margin_semantics": (
                "minimum normalized signed margin to the frozen POINT gates; "
                "it is an operating-point ranking diagnostic, not a probability"
            ),
            "orientation_gate": orientation_diagnostics,
            "original_evidence_mass": float(np.sum(evidence.base_weight)),
            "active_evidence_mass": float(np.sum(view.weight)),
            "active_mass_by_ring": active_mass,
            "remaining_fraction_by_ring": {
                evidence.ring_ids[index]: fraction
                for index, fraction in prepared.remaining_fraction.items()
            },
            "deleted_ring_ids": [
                evidence.ring_ids[index] for index in prepared.deleted_rings
            ],
            "weights_renormalised_after_deletion": False,
            "calibration_status": calibration_status,
            "calibration_frozen": calibration_frozen,
            "target_domain": target_domain,
            "model_risk_diagnostic": model_risk_diagnostics,
        }

        return LocateResult(
            crop_id=evidence.crop_id,
            state=state,
            raw_center_norm=_native_xy(raw_best.center),
            usable_center_norm=_native_xy(usable_center),
            direction_norm=_native_xy(direction),
            range_interval_norm=(
                [
                    float(distance_range[0]),
                    (
                        float(distance_range[1])
                        if distance_range[1] is not None
                        else None
                    ),
                ]
                if distance_range is not None
                else None
            ),
            objective=(
                float(raw_best.objective)
                if np.isfinite(raw_best.objective)
                else None
            ),
            radii_norm=radii,
            condition_ratio=float(common_seed.condition_ratio),
            weak_direction=_native_xy(common_seed.weak_direction),
            finite_modes=[self._mode_dict(item, evidence) for item in modes],
            far_scan=_json_finite(far_scan.as_dict()),
            profiles=_json_finite([profile.as_dict() for profile in profiles]),
            search_adequate=search_adequate,
            model_risk=model_risk,
            production_usable=production_usable,
            reason_codes=list(dict.fromkeys(reason_codes)),
            diagnostics=_json_finite(diagnostics),
            config_hash=self.config.sha256,
            run_id=self.run_id,
        )

    @staticmethod
    def _seed_diagnostics(seed: LinearSeedResult) -> dict[str, Any]:
        return _json_finite(
            {
                "center_norm": seed.center,
                "objective": seed.objective,
                "information": seed.information,
                "eigenvalues": seed.eigenvalues,
                "weak_direction_norm": seed.weak_direction,
                "strong_direction_norm": seed.strong_direction,
                "condition_ratio": seed.condition_ratio,
                "rank": seed.rank,
                "converged": seed.converged,
                "iterations": seed.iterations,
                "reason": seed.reason,
            }
        )
