"""Structured uncertainty replay for RAC-Pith.

This module is intentionally GT-blind.  It resamples complete parent rings or
applies correlated curve perturbations; individual quadrature nodes are never
treated as independent bootstrap observations.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .contracts import EvidenceBundle, GeometryState, LocateResult
from .contribution import perturb_evidence
from .estimator import RacPithEstimator


def _json_finite(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_finite(value.tolist())
    if isinstance(value, np.generic):
        return _json_finite(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return value


def ring_bootstrap_bundle(
    bundle: EvidenceBundle,
    sampled_ring_indices: np.ndarray,
) -> EvidenceBundle:
    """Duplicate full parent rings, re-indexing every draw and conserving budget.

    Each draw receives mass ``1 / n_draws``.  Within a draw, the source ring's
    original quadrature mass ratios are retained.  Duplicate draws get distinct
    nuisance-radius and arc IDs.  Ring order is disabled because resampling with
    replacement does not preserve a strict biological sequence.
    """

    sampled = np.asarray(sampled_ring_indices, dtype=np.int64).reshape(-1)
    if sampled.size == 0:
        raise ValueError("ring bootstrap requires at least one draw")
    if np.any(sampled < 0) or np.any(sampled >= len(bundle.ring_ids)):
        raise ValueError("sampled ring index is outside ring_ids")

    array_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "points_norm",
            "points_crop_px",
            "tangents",
            "sigma_x_norm",
            "sigma_alg_norm2",
            "base_weight",
            "ring_index",
            "arc_index",
            "source_s_px",
            "arc_fraction",
            "quality",
        )
    }
    ring_ids: list[str] = []
    arc_ids: list[str] = []
    for draw_index, source_ring_index in enumerate(sampled.tolist()):
        node_indices = np.flatnonzero(bundle.ring_index == source_ring_index)
        if node_indices.size == 0:
            raise ValueError(f"source ring {source_ring_index} has no evidence nodes")
        source_mass = bundle.base_weight[node_indices]
        source_mass_sum = float(source_mass.sum())
        if not math.isfinite(source_mass_sum) or source_mass_sum <= 0.0:
            raise ValueError("source ring has invalid evidence mass")
        ring_ids.append(f"bootstrap_{draw_index:03d}:{bundle.ring_ids[source_ring_index]}")

        local_arc_indices = np.empty(node_indices.size, dtype=np.int64)
        for source_arc_index in dict.fromkeys(bundle.arc_index[node_indices].tolist()):
            arc_id = f"bootstrap_{draw_index:03d}:{bundle.arc_ids[int(source_arc_index)]}"
            target_arc_index = len(arc_ids)
            arc_ids.append(arc_id)
            local_arc_indices[bundle.arc_index[node_indices] == source_arc_index] = target_arc_index

        for name in (
            "points_norm",
            "points_crop_px",
            "tangents",
            "sigma_x_norm",
            "sigma_alg_norm2",
            "source_s_px",
            "arc_fraction",
            "quality",
        ):
            array_parts[name].append(np.asarray(getattr(bundle, name)[node_indices]).copy())
        array_parts["base_weight"].append(
            source_mass / source_mass_sum / float(sampled.size)
        )
        array_parts["ring_index"].append(
            np.full(node_indices.size, draw_index, dtype=np.int64)
        )
        array_parts["arc_index"].append(local_arc_indices)

    metadata = dict(bundle.metadata)
    metadata.update(
        {
            "uncertainty_replay": "parent_ring_bootstrap",
            "bootstrap_source_ring_ids": [
                bundle.ring_ids[index] for index in sampled.tolist()
            ],
            "ring_order_reliable": False,
            "ring_order_disabled_reason": "bootstrap_with_replacement",
            "gt_available_to_estimator": False,
        }
    )
    result = replace(
        bundle,
        points_norm=np.concatenate(array_parts["points_norm"], axis=0),
        points_crop_px=np.concatenate(array_parts["points_crop_px"], axis=0),
        tangents=np.concatenate(array_parts["tangents"], axis=0),
        sigma_x_norm=np.concatenate(array_parts["sigma_x_norm"], axis=0),
        sigma_alg_norm2=np.concatenate(array_parts["sigma_alg_norm2"], axis=0),
        base_weight=np.concatenate(array_parts["base_weight"], axis=0),
        ring_index=np.concatenate(array_parts["ring_index"], axis=0),
        arc_index=np.concatenate(array_parts["arc_index"], axis=0),
        source_s_px=np.concatenate(array_parts["source_s_px"], axis=0),
        arc_fraction=np.concatenate(array_parts["arc_fraction"], axis=0),
        quality=np.concatenate(array_parts["quality"], axis=0),
        ring_ids=tuple(ring_ids),
        arc_ids=tuple(arc_ids),
        metadata=metadata,
    )
    result.validate()
    return result


def _record_result(kind: str, index: int, result: LocateResult) -> dict[str, Any]:
    return {
        "replay_kind": kind,
        "replay_index": index,
        "status": "FINISHED",
        "state": result.state.value,
        "raw_center_norm": result.raw_center_norm,
        "search_adequate": result.search_adequate,
        "finite_mode_count": len(result.finite_modes),
        "reason_codes": list(result.reason_codes),
    }


def _failure_record(kind: str, index: int, reason: str) -> dict[str, Any]:
    return {
        "replay_kind": kind,
        "replay_index": index,
        "status": "FAIL",
        "state": "REJECT",
        "raw_center_norm": None,
        "search_adequate": False,
        "finite_mode_count": 0,
        "reason_codes": [reason],
    }


def _ellipse(
    centers: np.ndarray,
    confidence_level: float,
    minimum_samples: int,
) -> dict[str, Any]:
    if len(centers) < minimum_samples:
        return {
            "valid": False,
            "reason": "TOO_FEW_POINT_REPLAYS",
            "samples": int(len(centers)),
        }
    covariance = np.cov(centers, rowvar=False, ddof=1)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return {
            "valid": False,
            "reason": "NONFINITE_COVARIANCE",
            "samples": int(len(centers)),
        }
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    # Chi-square(2) CDF is 1-exp(-q/2), so no extra statistical dependency is
    # needed for the registered two-dimensional Gaussian reference ellipse.
    chi_square_quantile = -2.0 * math.log(1.0 - confidence_level)
    semi_axes = np.sqrt(chi_square_quantile * eigenvalues)
    major_vector = eigenvectors[:, 0]
    return {
        "valid": True,
        "interpretation": (
            "empirical stability ellipse under registered structured replay; "
            "not a calibrated confidence region until independent-tree coverage passes"
        ),
        "samples": int(len(centers)),
        "confidence_reference": float(confidence_level),
        "center_median_norm": np.median(centers, axis=0),
        "center_mean_norm": np.mean(centers, axis=0),
        "covariance_norm2": covariance,
        "major_semi_axis_norm": float(semi_axes[0]),
        "minor_semi_axis_norm": float(semi_axes[1]),
        "major_axis_angle_rad": float(math.atan2(major_vector[1], major_vector[0])),
    }


def run_structured_uncertainty(
    bundle: EvidenceBundle,
    estimator: RacPithEstimator,
    baseline: LocateResult,
    *,
    ring_replays: int,
    curve_replays: int,
    seed: int,
    curve_correlation_fraction: float,
    confidence_level: float,
    minimum_point_samples: int,
    minimum_point_fraction: float,
) -> dict[str, Any]:
    """Run GT-blind ring and curve replays and issue a conservative sidecar gate."""

    if baseline.crop_id != bundle.crop_id:
        raise ValueError("baseline/evidence crop_id mismatch")
    if baseline.config_hash != estimator.config.sha256:
        raise ValueError("baseline/uncertainty config hash mismatch")
    if ring_replays < 0 or curve_replays < 0 or ring_replays + curve_replays == 0:
        raise ValueError("at least one non-negative structured replay count is required")
    if not 0.0 < curve_correlation_fraction <= 1.0:
        raise ValueError("curve_correlation_fraction must lie in (0,1]")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0,1)")
    if minimum_point_samples < 2:
        raise ValueError("minimum_point_samples must be at least two")
    if not 0.0 <= minimum_point_fraction <= 1.0:
        raise ValueError("minimum_point_fraction must lie in [0,1]")

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    minimum_unique = int(estimator.solver["min_parent_rings"])
    ring_count = len(bundle.ring_ids)
    for replay_index in range(ring_replays):
        sampled = rng.integers(0, ring_count, size=ring_count, endpoint=False)
        if len(np.unique(sampled)) < minimum_unique:
            records.append(
                _failure_record(
                    "parent_ring_bootstrap",
                    replay_index,
                    "BOOTSTRAP_TOO_FEW_UNIQUE_SOURCE_RINGS",
                )
            )
            continue
        try:
            resampled = ring_bootstrap_bundle(bundle, sampled)
            result = estimator.fit(resampled)
            records.append(_record_result("parent_ring_bootstrap", replay_index, result))
        except Exception as exc:
            records.append(
                _failure_record(
                    "parent_ring_bootstrap",
                    replay_index,
                    f"{type(exc).__name__}: {exc}",
                )
            )

    for replay_index in range(curve_replays):
        try:
            perturbed = perturb_evidence(bundle, rng, curve_correlation_fraction)
            result = estimator.fit(perturbed)
            records.append(_record_result("correlated_curve", replay_index, result))
        except Exception as exc:
            records.append(
                _failure_record(
                    "correlated_curve",
                    replay_index,
                    f"{type(exc).__name__}: {exc}",
                )
            )

    requested = ring_replays + curve_replays
    point_records = [
        row
        for row in records
        if row["status"] == "FINISHED"
        and row["state"] == "POINT"
        and row["raw_center_norm"] is not None
    ]
    centers = (
        np.asarray([row["raw_center_norm"] for row in point_records], dtype=np.float64)
        if point_records
        else np.empty((0, 2), dtype=np.float64)
    )
    ellipse = _ellipse(centers, confidence_level, minimum_point_samples)
    state_counts = {
        state.value: sum(row["state"] == state.value for row in records)
        for state in GeometryState
    }
    baseline_center = (
        np.asarray(baseline.raw_center_norm, dtype=np.float64)
        if baseline.raw_center_norm is not None
        else None
    )
    tail_trigger = float(estimator.search["far_distance_trigger_norm"])
    replay_norms = np.linalg.norm(centers, axis=1) if len(centers) else np.empty(0)
    explicit_infinite_replay = any(
        row["status"] == "FINISHED" and row["state"] in {"RAY", "AXIS"}
        for row in records
    )
    new_long_tail = bool(
        explicit_infinite_replay
        or (
            len(replay_norms)
            and np.any(
                replay_norms
                > max(
                    tail_trigger,
                    (
                        float(np.linalg.norm(baseline_center))
                        if baseline_center is not None
                        else 0.0
                    )
                    + float(estimator.search["profile_range_span_norm"]),
                )
            )
        )
    )
    new_multimodal = bool(
        any(
            row["state"] == "MULTIMODAL"
            or int(row.get("finite_mode_count", 0)) > len(baseline.finite_modes)
            for row in records
            if row["status"] == "FINISHED"
        )
    )
    point_fraction = len(point_records) / requested
    adjudicated_state = baseline.state.value
    gate_reasons: list[str] = []
    requires_search_reaudit = False
    if baseline.state == GeometryState.POINT:
        if new_long_tail or new_multimodal:
            adjudicated_state = "REJECT"
            requires_search_reaudit = True
            gate_reasons.append("STRUCTURED_REPLAY_REVEALED_NEW_TAIL_OR_MODE")
        elif not ellipse.get("valid", False):
            adjudicated_state = "REJECT"
            gate_reasons.append("STRUCTURED_UNCERTAINTY_UNRESOLVED")
        elif point_fraction < minimum_point_fraction:
            adjudicated_state = "RANGE"
            gate_reasons.append("POINT_STATE_REPLAY_FRACTION_TOO_LOW")
        elif float(ellipse["major_semi_axis_norm"]) > float(
            estimator.state_config["point_bootstrap_major_axis_norm"]
        ):
            adjudicated_state = "RANGE"
            gate_reasons.append("POINT_REPLAY_ELLIPSE_TOO_WIDE")

    result = {
        "schema_version": "racpith.uncertainty.v1",
        "crop_id": bundle.crop_id,
        "tree_id": bundle.tree_id,
        "section_id": bundle.section_id,
        "config_hash": estimator.config.sha256,
        "baseline_run_id": baseline.run_id,
        "gt_available_to_replay": False,
        "node_iid_bootstrap_used": False,
        "requested": {
            "parent_ring_bootstrap": ring_replays,
            "correlated_curve": curve_replays,
        },
        "replay_protocol": {
            "random_seed": int(seed),
            "curve_correlation_fraction": float(curve_correlation_fraction),
            "ellipse_confidence_reference": float(confidence_level),
            "minimum_point_samples": int(minimum_point_samples),
            "minimum_point_fraction": float(minimum_point_fraction),
        },
        "completed": sum(row["status"] == "FINISHED" for row in records),
        "failed": sum(row["status"] != "FINISHED" for row in records),
        "state_counts": state_counts,
        "point_fraction_of_all_requested": point_fraction,
        "point_centers_norm": centers,
        "ellipse": ellipse,
        "new_long_tail": new_long_tail,
        "explicit_infinite_replay": explicit_infinite_replay,
        "new_multimodal_support": new_multimodal,
        "requires_search_reaudit": requires_search_reaudit,
        "baseline_state": baseline.state.value,
        "adjudicated_state": adjudicated_state,
        "gate_reason_codes": gate_reasons,
        "replays": records,
    }
    return _json_finite(result)
