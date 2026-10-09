"""Low-capacity, GT-blind diagnostics for common-circle model mismatch."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .numerics import EvidenceView


def assess_model_risk(
    view: EvidenceView,
    center: np.ndarray,
    radii_by_ring: Mapping[int, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Detect coherent first-harmonic residual structure without moving the centre.

    This is a diagnostic, not a bias-correction model.  Rings with insufficient
    angular support or an ill-conditioned harmonic design are excluded and
    reported explicitly.
    """

    minimum_coverage = math.radians(float(config["minimum_angular_coverage_deg"]))
    maximum_condition = float(config["maximum_harmonic_condition"])
    minimum_rings = int(config["minimum_analyzable_rings"])
    ring_rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    standardized_amplitudes: list[float] = []
    for ring in view.active_rings:
        mask = view.ring_index == ring
        radius = radii_by_ring.get(int(ring))
        if radius is None or not math.isfinite(float(radius)) or np.count_nonzero(mask) < 6:
            ring_rows.append({"ring_index": int(ring), "status": "INSUFFICIENT_OR_NO_RADIUS"})
            continue
        displacement = view.points[mask] - center[None, :]
        distance = np.linalg.norm(displacement, axis=1)
        angle = np.mod(np.arctan2(displacement[:, 1], displacement[:, 0]), 2.0 * np.pi)
        ordered = np.sort(angle)
        gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
        angular_coverage = float(2.0 * np.pi - np.max(gaps))
        if angular_coverage < minimum_coverage:
            ring_rows.append(
                {
                    "ring_index": int(ring),
                    "status": "ANGULAR_SUPPORT_TOO_NARROW",
                    "angular_coverage_deg": math.degrees(angular_coverage),
                }
            )
            continue
        design = np.column_stack([np.ones(len(angle)), np.cos(angle), np.sin(angle)])
        precision = view.weight[mask] / np.square(view.sigma_x[mask])
        normal = design.T @ (precision[:, None] * design)
        condition = float(np.linalg.cond(normal))
        if not math.isfinite(condition) or condition > maximum_condition:
            ring_rows.append(
                {
                    "ring_index": int(ring),
                    "status": "HARMONIC_DESIGN_ILL_CONDITIONED",
                    "condition": condition,
                    "angular_coverage_deg": math.degrees(angular_coverage),
                }
            )
            continue
        residual = distance - float(radius)
        rhs = design.T @ (precision * residual)
        coefficient = np.linalg.solve(normal, rhs)
        vector = np.asarray(coefficient[1:3], dtype=np.float64)
        amplitude = float(np.linalg.norm(vector))
        sigma_reference = float(np.median(view.sigma_x[mask]))
        standardized = amplitude / max(sigma_reference, 1e-12)
        vectors.append(vector)
        standardized_amplitudes.append(standardized)
        ring_rows.append(
            {
                "ring_index": int(ring),
                "status": "ANALYZED",
                "angular_coverage_deg": math.degrees(angular_coverage),
                "condition": condition,
                "first_harmonic_vector_norm": vector.tolist(),
                "first_harmonic_amplitude_norm": amplitude,
                "standardized_amplitude": standardized,
            }
        )

    ordered_radii = [
        float(radii_by_ring[ring])
        for ring in sorted(radii_by_ring)
        if math.isfinite(float(radii_by_ring[ring]))
    ]
    order_violation_fraction = (
        float(np.mean(np.diff(ordered_radii) <= 0.0)) if len(ordered_radii) >= 2 else None
    )
    if len(vectors) < minimum_rings:
        return {
            "level": "UNKNOWN",
            "reason_codes": ["TOO_FEW_RINGS_FOR_MODEL_RISK"],
            "analyzable_rings": len(vectors),
            "ring_diagnostics": ring_rows,
            "order_violation_fraction": order_violation_fraction,
        }

    amplitudes = np.asarray(standardized_amplitudes, dtype=np.float64)
    unit_vectors = np.asarray(
        [vector / max(np.linalg.norm(vector), 1e-12) for vector in vectors]
    )
    coherence = float(np.linalg.norm(np.mean(unit_vectors, axis=0)))
    median_amplitude = float(np.median(amplitudes))
    high = bool(
        (
            median_amplitude >= float(config["high_standardized_amplitude"])
            and coherence >= float(config["high_directional_coherence"])
        )
        or (
            order_violation_fraction is not None
            and order_violation_fraction >= float(config["high_order_violation_fraction"])
        )
    )
    low = bool(
        median_amplitude <= float(config["low_standardized_amplitude"])
        and (
            order_violation_fraction is None
            or order_violation_fraction <= float(config["low_order_violation_fraction"])
        )
    )
    level = "HIGH_RISK" if high else "LOW_RISK" if low else "MEDIUM_RISK"
    reason_codes = []
    if high and median_amplitude >= float(config["high_standardized_amplitude"]):
        reason_codes.append("COHERENT_LOW_FREQUENCY_RESIDUAL")
    if high and order_violation_fraction is not None and order_violation_fraction >= float(
        config["high_order_violation_fraction"]
    ):
        reason_codes.append("SYSTEMATIC_RING_ORDER_VIOLATION")
    return {
        "level": level,
        "reason_codes": reason_codes,
        "analyzable_rings": len(vectors),
        "median_standardized_first_harmonic_amplitude": median_amplitude,
        "directional_coherence": coherence,
        "order_violation_fraction": order_violation_fraction,
        "ring_diagnostics": ring_rows,
        "coordinate_changed_by_risk_diagnostic": False,
    }
