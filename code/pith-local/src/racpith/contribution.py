from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import EvidenceBundle, GeometryState, LocateResult
from .estimator import RacPithEstimator
from .groups import DeletionGroup
from .numerics import EvidenceView, VarProProblem
from .provenance import sha256_file


def _result_digest(result: LocateResult) -> str:
    payload = json.dumps(
        result.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _view(bundle: EvidenceBundle, active: np.ndarray) -> EvidenceView:
    return EvidenceView(
        points=np.asarray(bundle.points_norm[active], dtype=np.float64),
        tangents=np.asarray(bundle.tangents[active], dtype=np.float64),
        sigma_x=np.asarray(bundle.sigma_x_norm[active], dtype=np.float64),
        sigma_alg=np.asarray(bundle.sigma_alg_norm2[active], dtype=np.float64),
        weight=np.asarray(bundle.base_weight[active], dtype=np.float64),
        ring_index=np.asarray(bundle.ring_index[active], dtype=np.int64),
    )


def _problem(bundle: EvidenceBundle, active: np.ndarray, estimator: RacPithEstimator) -> VarProProblem:
    return VarProProblem(
        _view(bundle, active),
        loss=estimator.loss,
        delta=estimator.delta,
        radius_tolerance=float(estimator.solver["radius_tol"]),
        radius_max_iterations=int(estimator.solver["radius_max_iter"]),
    )


def _directional_information(
    bundle: EvidenceBundle, remove_mask: np.ndarray
) -> tuple[float | None, str | None]:
    tangent = bundle.tangents
    mass = bundle.base_weight
    full = tangent.T @ (mass[:, None] * tangent)
    remaining = ~remove_mask
    minus = tangent[remaining].T @ (mass[remaining, None] * tangent[remaining])
    epsilon = max(float(np.trace(full)), 1.0) * 1e-10
    sign_full, log_full = np.linalg.slogdet(full + epsilon * np.eye(2))
    sign_minus, log_minus = np.linalg.slogdet(minus + epsilon * np.eye(2))
    if sign_full <= 0 or sign_minus <= 0:
        return None, "DIRECTION_INFO_SINGULAR"
    value = float(log_full - log_minus)
    if value < -1e-8:
        return value, "DIRECTION_INFO_NEGATIVE"
    return value, None


def _support_widths(result: LocateResult) -> tuple[float | None, float | None]:
    center = (
        np.asarray(result.raw_center_norm, dtype=np.float64)
        if result.raw_center_norm is not None
        else None
    )
    local = result.diagnostics.get("local_support", {})
    major = local.get("major_span") if isinstance(local, Mapping) else None
    direction_width = None
    if center is not None and major is not None and np.isfinite(float(major)):
        direction_width = float(
            2.0 * np.arctan2(0.5 * float(major), max(np.linalg.norm(center), 1e-12))
        )
    elif result.far_scan:
        supported = result.far_scan.get("supported_minima", [])
        angles = [float(item["phi_rad"]) for item in supported if item.get("phi_rad") is not None]
        if len(angles) == 1:
            direction_width = float(result.far_scan.get("angle_step_rad", 0.0))
        elif len(angles) > 1:
            doubled = np.exp(2j * np.asarray(angles))
            concentration = abs(np.mean(doubled))
            direction_width = float(np.sqrt(max(0.0, -2.0 * np.log(max(concentration, 1e-12)))) / 2.0)

    range_width = None
    if result.range_interval_norm is not None:
        lower, upper = result.range_interval_norm
        if lower is not None and upper is not None:
            range_width = float(np.log((float(upper) + 1e-9) / (float(lower) + 1e-9)))
    elif center is not None and major is not None and np.isfinite(float(major)):
        radius = float(np.linalg.norm(center))
        lo = max(0.0, radius - 0.5 * float(major))
        hi = radius + 0.5 * float(major)
        range_width = float(np.log((hi + 1e-9) / (lo + 1e-9)))
    return direction_width, range_width


def _witness_loss(
    bundle: EvidenceBundle,
    witness_mask: np.ndarray,
    center: np.ndarray,
    estimator: RacPithEstimator,
) -> float:
    return float(_problem(bundle, witness_mask, estimator).objective(center))


def cross_fitted_contribution(
    bundle: EvidenceBundle,
    group: DeletionGroup,
    estimator: RacPithEstimator,
    config: Mapping[str, Any],
) -> tuple[float | None, list[float], str | None]:
    target_ring = bundle.ring_ids.index(group.ring_id)
    witness_rings = [index for index in range(len(bundle.ring_ids)) if index != target_ring]
    minimum = int(config["minimum_crossfit_witness_rings"])
    if len(witness_rings) < minimum:
        return None, [], "INSUFFICIENT_INDEPENDENT_WITNESS_RINGS"
    fold_count = min(int(config["crossfit_folds"]), len(witness_rings))
    folds = [witness_rings[index::fold_count] for index in range(fold_count)]
    values: list[float] = []
    for witness in folds:
        witness_mask = np.isin(bundle.ring_index, witness)
        tangent = bundle.tangents[witness_mask]
        mass = bundle.base_weight[witness_mask]
        information = tangent.T @ (mass[:, None] * tangent)
        eigenvalues = np.linalg.eigvalsh(information)
        ratio = float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else 0.0
        if ratio < float(estimator.search["condition_ratio_trigger"]):
            continue
        train_plus = ~witness_mask
        train_minus = train_plus & ~group.remove_mask
        plus_result = estimator.fit(bundle, train_plus)
        minus_result = estimator.fit(bundle, train_minus)
        if plus_result.state != GeometryState.POINT or minus_result.state != GeometryState.POINT:
            continue
        plus_center = np.asarray(plus_result.raw_center_norm, dtype=np.float64)
        minus_center = np.asarray(minus_result.raw_center_norm, dtype=np.float64)
        improvement = _witness_loss(bundle, witness_mask, minus_center, estimator) - _witness_loss(
            bundle, witness_mask, plus_center, estimator
        )
        if np.isfinite(improvement):
            values.append(float(improvement))
    if len(values) < max(2, (fold_count + 1) // 2):
        return None, values, "CROSSFIT_FOLDS_UNRESOLVED"
    return float(np.median(values)), values, None


def perturb_evidence(
    bundle: EvidenceBundle,
    rng: np.random.Generator,
    correlation_fraction: float,
) -> EvidenceBundle:
    points = bundle.points_norm.copy()
    tangents = bundle.tangents.copy()
    for arc_index in range(len(bundle.arc_ids)):
        indices = np.flatnonzero(bundle.arc_index == arc_index)
        if len(indices) < 3:
            continue
        order = np.argsort(bundle.source_s_px[indices])
        ordered = indices[order]
        raw = rng.normal(size=len(ordered))
        window = max(1, int(round(correlation_fraction * len(ordered))))
        kernel = np.ones(window, dtype=np.float64) / window
        correlated = np.convolve(raw, kernel, mode="same")
        standard = float(np.std(correlated))
        if standard > 1e-12:
            correlated /= standard
        normal = np.column_stack([-tangents[ordered, 1], tangents[ordered, 0]])
        displacement = correlated * bundle.sigma_x_norm[ordered]
        points[ordered] += displacement[:, None] * normal
        derivative = np.gradient(points[ordered], axis=0)
        norm = np.linalg.norm(derivative, axis=1)
        stable = norm > 1e-12
        tangents[ordered[stable]] = derivative[stable] / norm[stable, None]
    width, height = bundle.crop_size_px
    crop_center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    points_crop_px = crop_center + points * bundle.normalization_scale_px
    # The algebraic seed scale is a first-order propagation bound tied to the
    # perturbed point location.  Keeping the old value would make replay use an
    # internally inconsistent evidence bundle.
    sigma_alg = np.maximum(
        2.0 * (np.linalg.norm(points, axis=1) + 1.0) * bundle.sigma_x_norm,
        1.0e-12,
    )
    result = replace(
        bundle,
        points_norm=points,
        points_crop_px=points_crop_px,
        tangents=tangents,
        sigma_alg_norm2=sigma_alg,
    )
    result.validate()
    return result


def _paired_replay_values(
    bundle: EvidenceBundle,
    groups: list[DeletionGroup],
    estimator: RacPithEstimator,
    gt_norm: np.ndarray,
    count: int,
    seed: int,
    correlation_fraction: float,
    gt_sigma_norm: float,
) -> dict[str, list[float]]:
    values = {group.group_id: [] for group in groups}
    rng = np.random.default_rng(seed)
    for _ in range(count):
        perturbed = perturb_evidence(bundle, rng, correlation_fraction)
        perturbed_gt = gt_norm + rng.normal(scale=gt_sigma_norm, size=2)
        full = estimator.fit(perturbed)
        if full.state != GeometryState.POINT or full.raw_center_norm is None:
            continue
        full_center = np.asarray(full.raw_center_norm, dtype=np.float64)
        full_error = float(np.linalg.norm(full_center - perturbed_gt))
        for group in groups:
            minus = estimator.fit(perturbed, ~group.remove_mask)
            if minus.state != GeometryState.POINT or minus.raw_center_norm is None:
                continue
            minus_center = np.asarray(minus.raw_center_norm, dtype=np.float64)
            values[group.group_id].append(
                float(np.linalg.norm(minus_center - perturbed_gt) - full_error)
            )
    return values


def exact_delete_refit(
    bundle: EvidenceBundle,
    estimator: RacPithEstimator,
    groups: Iterable[DeletionGroup],
    baseline_result: LocateResult,
    baseline_result_path: str | Path,
    gt_norm: np.ndarray | None,
    *,
    replay_count: int | None = None,
    replay_seed: int = 20260906,
    replay_correlation_fraction: float = 0.05,
    gt_sigma_px: float | None = None,
    compute_crossfit: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, LocateResult]]:
    bundle.validate()
    if baseline_result.crop_id != bundle.crop_id:
        raise ValueError("baseline/evidence crop_id mismatch")
    if baseline_result.config_hash != estimator.config.sha256:
        raise ValueError("baseline/delete-refit config hash mismatch")
    contribution_cfg = estimator.config.section("contribution")
    group_list = list(groups)
    count = int(contribution_cfg.get("replay_count", 0) if replay_count is None else replay_count)
    if count < 0:
        raise ValueError("replay_count must be non-negative")
    if not 0.0 < float(replay_correlation_fraction) <= 1.0:
        raise ValueError("replay_correlation_fraction must lie in (0,1]")
    gt_array = np.asarray(gt_norm, dtype=np.float64).reshape(2) if gt_norm is not None else None
    if gt_array is not None and not np.all(np.isfinite(gt_array)):
        raise ValueError("gt_norm must be finite when supplied")
    gt_sigma = float(
        contribution_cfg.get("gt_sigma_px", 0.0) if gt_sigma_px is None else gt_sigma_px
    ) / bundle.normalization_scale_px
    if not np.isfinite(gt_sigma) or gt_sigma < 0.0:
        raise ValueError("gt_sigma_px must be finite and non-negative")
    replay: dict[str, list[float]] = {group.group_id: [] for group in group_list}
    if gt_array is not None and count > 0:
        replay = _paired_replay_values(
            bundle,
            group_list,
            estimator,
            gt_array,
            count,
            replay_seed,
            replay_correlation_fraction,
            gt_sigma,
        )

    baseline_path = Path(baseline_result_path)
    baseline_sha = sha256_file(baseline_path)
    baseline_center = (
        np.asarray(baseline_result.raw_center_norm, dtype=np.float64)
        if baseline_result.raw_center_norm is not None
        else None
    )
    epsilon = float(contribution_cfg["epsilon_norm"])
    records: list[dict[str, Any]] = []
    minus_results: dict[str, LocateResult] = {}
    for group in group_list:
        active = ~group.remove_mask
        minus = estimator.fit(bundle, active)
        minus_results[group.group_id] = minus
        minus_center = (
            np.asarray(minus.raw_center_norm, dtype=np.float64)
            if minus.raw_center_norm is not None
            else None
        )
        roles: list[str] = []
        if baseline_result.state == GeometryState.POINT and minus.state != GeometryState.POINT:
            roles.append("IDENTIFIABILITY_CRITICAL")
            if minus.state in {GeometryState.RAY, GeometryState.AXIS}:
                roles.extend(["DIRECTION_CRITICAL", "RANGE_CRITICAL"])
            elif minus.state == GeometryState.RANGE:
                roles.append("RANGE_CRITICAL")
            elif minus.state == GeometryState.MULTIMODAL:
                roles.append("MODE_EXCLUSION")
        if len(minus.finite_modes) > len(baseline_result.finite_modes):
            roles.append("MODE_EXCLUSION")
        full_direction_width, full_range_width = _support_widths(baseline_result)
        minus_direction_width, minus_range_width = _support_widths(minus)
        direction_width_log_ratio = (
            float(np.log((minus_direction_width + 1e-9) / (full_direction_width + 1e-9)))
            if full_direction_width is not None and minus_direction_width is not None
            else None
        )
        range_width_log_ratio = (
            float(np.log((minus_range_width + 1e-9) / (full_range_width + 1e-9)))
            if full_range_width is not None and minus_range_width is not None
            else None
        )

        shift_vector: np.ndarray | None = None
        shift = None
        if baseline_center is not None and minus_center is not None:
            shift_vector = baseline_center - minus_center
            shift = float(np.linalg.norm(shift_vector))
            if shift >= float(contribution_cfg["high_leverage_norm"]):
                roles.append("HIGH_LEVERAGE")

        conflict = None
        conflict_error = None
        if baseline_center is not None and minus_center is not None:
            try:
                minus_problem = _problem(bundle, active, estimator)
                conflict = float(
                    minus_problem.objective(baseline_center)
                    - minus_problem.objective(minus_center)
                )
                if conflict >= float(contribution_cfg["conflict_threshold"]):
                    roles.append("CONFLICTING")
                if conflict < -max(1e-8, float(estimator.solver["loss_equivalence"])):
                    conflict_error = "NEGATIVE_CONFLICT_REFIT_INCONSISTENT"
            except Exception as exc:
                conflict_error = f"{type(exc).__name__}: {exc}"

        information, information_error = _directional_information(bundle, group.remove_mask)
        if information is not None and information >= float(
            contribution_cfg["direction_info_threshold"]
        ):
            roles.append("DIRECTION_CRITICAL")

        c_norm = c_sq = c_sq_identity = c_rel = None
        gt_point_label: str | None = None
        gt_label: str | None = None
        ci: list[float] | None = None
        gt_abstain: str | None = None
        if (
            gt_array is not None
            and baseline_result.state == GeometryState.POINT
            and minus.state == GeometryState.POINT
            and baseline_center is not None
            and minus_center is not None
        ):
            error_all = float(np.linalg.norm(baseline_center - gt_array))
            error_minus = float(np.linalg.norm(minus_center - gt_array))
            c_norm = error_minus - error_all
            c_sq = error_minus**2 - error_all**2
            delta_center = baseline_center - minus_center
            c_sq_identity = float(
                2.0 * np.dot(delta_center, gt_array - minus_center)
                - np.dot(delta_center, delta_center)
            )
            c_rel = float(c_sq / (error_minus**2 + error_all**2 + epsilon**2))
            if c_norm > epsilon:
                gt_point_label = "BENEFICIAL_GT"
            elif c_norm < -epsilon:
                gt_point_label = "HARMFUL_GT"
            else:
                gt_point_label = "NEUTRAL"
            replay_values = np.asarray(replay[group.group_id], dtype=np.float64)
            if count > 0 and len(replay_values) >= max(10, int(np.ceil(0.6 * count))):
                lower, upper = np.quantile(replay_values, [0.025, 0.975])
                ci = [float(lower), float(upper)]
                if lower > epsilon:
                    gt_label = "BENEFICIAL_GT"
                elif upper < -epsilon:
                    gt_label = "HARMFUL_GT"
                elif lower >= -epsilon and upper <= epsilon:
                    gt_label = "NEUTRAL"
                else:
                    gt_label = "UNCERTAIN"
            else:
                gt_label = "UNCERTAIN"
                gt_abstain = "PAIRED_REPLAY_CI_UNAVAILABLE"
        elif gt_array is not None:
            gt_abstain = "FULL_OR_MINUS_NOT_POINT"

        cf_value = None
        cf_folds: list[float] = []
        cf_abstain = None
        if compute_crossfit:
            cf_value, cf_folds, cf_abstain = cross_fitted_contribution(
                bundle, group, estimator, contribution_cfg
            )
        if (
            shift is not None
            and shift < epsilon
            and (conflict is None or abs(conflict) < float(contribution_cfg["conflict_threshold"]))
            and (information is None or information < float(contribution_cfg["direction_info_threshold"]))
        ):
            roles.append("REDUNDANT")

        record = {
            "schema_version": "racpith.contribution.v1",
            "crop_id": bundle.crop_id,
            "tree_id": bundle.tree_id,
            "section_id": bundle.section_id,
            "group_id": group.group_id,
            "level": group.level,
            "ring_id": group.ring_id,
            "arc_id": group.arc_id,
            "partition_id": group.partition_id,
            "scale_fraction": group.scale_fraction,
            "phase": group.phase,
            "interval_fraction": list(group.interval_fraction)
            if group.interval_fraction is not None
            else None,
            "removed_mass": group.removed_mass,
            "deletion_mask_sha256": group.mask_hash,
            "baseline_result_sha256": baseline_sha,
            "baseline_result_digest": _result_digest(baseline_result),
            "minus_result_digest": _result_digest(minus),
            "frozen_config_hash": estimator.config.sha256,
            "full_state": baseline_result.state.value,
            "minus_state": minus.state.value,
            "minus_search_adequate": minus.search_adequate,
            "shift_vector_norm": shift_vector.tolist() if shift_vector is not None else None,
            "shift_norm": shift,
            "shift_px": shift * bundle.normalization_scale_px if shift is not None else None,
            "conflict_cost": conflict,
            "conflict_error": conflict_error,
            "directional_info_gain": information,
            "directional_info_error": information_error,
            "full_direction_width_rad": full_direction_width,
            "minus_direction_width_rad": minus_direction_width,
            "direction_width_log_ratio": direction_width_log_ratio,
            "full_range_log_width": full_range_width,
            "minus_range_log_width": minus_range_width,
            "range_width_log_ratio": range_width_log_ratio,
            "mode_count_change": len(minus.finite_modes) - len(baseline_result.finite_modes),
            "contrib_gt_norm": c_norm,
            "contrib_gt_px": c_norm * bundle.normalization_scale_px if c_norm is not None else None,
            "contrib_gt_sq": c_sq,
            "contrib_gt_sq_identity": c_sq_identity,
            "contrib_gt_rel": c_rel,
            "gt_point_label": gt_point_label,
            "contrib_gt_ci_norm": ci,
            "gt_label": gt_label,
            "gt_abstain_reason": gt_abstain,
            "contrib_cf": cf_value,
            "contrib_cf_folds": cf_folds,
            "cf_abstain_reason": cf_abstain,
            "image_quality_mean": float(np.average(bundle.quality[group.remove_mask], weights=bundle.base_weight[group.remove_mask])),
            "roles": list(dict.fromkeys(roles)),
            "replay_requested": count,
            "replay_eligible": len(replay[group.group_id]),
            "all_arc_coordinate_changed": False,
            "abstain_reason": gt_abstain if gt_array is not None else cf_abstain,
        }
        records.append(record)
    return records, minus_results
