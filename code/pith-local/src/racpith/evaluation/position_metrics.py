"""Four-way pith-position grouping and three-layer error metrics (shusui.txt).

Layers (all derived from the same GT->prediction displacement, in px):
1. ``error_px``            — absolute pixel error;
2. ``E_RF_pct``            — 100 * error / R_F, R_F = D_F/2 of the FULL parent
                             section (frozen, shared by all crops of a section);
3. ``E_w`` / ``E_wmax``    — error / w_ref and error / w_max (frozen adjacent
                             ring spacings; E_w is NOT a tree-age error);
4. ``E_mm``                — empty (``MM_UNAVAILABLE_NO_CALIBRATION``) because
                             mm_per_pixel is absent dataset-wide; never guessed.

Groups: IN_IMAGE / NEAR (d/R_F <= 0.35) / MID (<= 0.70) / FAR (> 0.70) where d
is the crop-center to pith distance. Thresholds are frozen in the eval config.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

POSITION_METRICS_SCHEMA = "racpith.position_metrics.v1"

NEAR_RATIO = 0.35
MID_RATIO = 0.70
GROUP_ORDER = ["IN_IMAGE", "NEAR", "MID", "FAR"]
METRIC_KEYS = ["error_px", "E_RF_pct", "E_w", "E_wmax"]


def classify_pith_group(
    pith_px: Sequence[float],
    image_size: Sequence[float],
    distance_to_pith_px: float | None,
    r_f_px: float,
) -> tuple[str, bool, float]:
    """Return (group, inside, d_over_RF)."""
    width, height = (float(v) for v in image_size)
    x, y = (float(v) for v in pith_px)
    inside = 0.0 <= x < width and 0.0 <= y < height
    if inside:
        ratio = 0.0
        return "IN_IMAGE", True, ratio
    if distance_to_pith_px is None:
        distance = float(np.hypot(x - width / 2.0, y - height / 2.0))
    else:
        distance = float(distance_to_pith_px)
    ratio = distance / r_f_px if r_f_px > 0 else float("inf")
    if ratio <= NEAR_RATIO:
        return "NEAR", False, ratio
    if ratio <= MID_RATIO:
        return "MID", False, ratio
    return "FAR", False, ratio


def build_position_metrics(
    manifest_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    scales: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """One row per manifest crop with group and all three metric layers."""
    records: list[dict[str, Any]] = []
    for row in manifest_rows:
        crop_id = str(row["crop_id"])
        section_id = str(row["section_id"])
        scale = scales.get(section_id)
        gt = np.asarray(row["pith_px"], dtype=np.float64)
        r_f = float(scale["R_F_px"]) if scale else float("nan")
        w_ref = float(scale["w_ref_px"]) if scale and scale.get("w_ref_px") is not None else float("nan")
        w_max = float(scale["w_max_px"]) if scale and scale.get("w_max_px") is not None else float("nan")
        group, inside, ratio = classify_pith_group(
            gt, row["image_size"], row.get("metadata", {}).get("distance_to_pith_px"), r_f
        )
        pred = predictions.get(crop_id)
        pred_pt = pred.get("pith_px") if pred else None
        record: dict[str, Any] = {
            "crop_id": crop_id,
            "tree_id": str(row["tree_id"]),
            "section_id": section_id,
            "group": group,
            "pith_inside_crop": inside,
            "d_over_RF": None if inside else ratio,
            "geometry_state": pred.get("geometry_state") if pred else "NO_FINAL_ROW",
            "has_prediction": pred_pt is not None,
            "mm_per_pixel": row.get("mm_per_pixel"),
            "R_F_px": r_f,
            "w_ref_px": w_ref,
            "w_max_px": w_max,
            "error_px": None,
            "E_RF_pct": None,
            "E_w": None,
            "E_wmax": None,
            "E_mm": None,
            "E_mm_status": "MM_UNAVAILABLE_NO_CALIBRATION",
        }
        if pred_pt is not None:
            err = float(np.linalg.norm(np.asarray(pred_pt, dtype=np.float64) - gt))
            record["error_px"] = err
            record["E_RF_pct"] = 100.0 * err / r_f if np.isfinite(r_f) and r_f > 0 else None
            record["E_w"] = err / w_ref if np.isfinite(w_ref) and w_ref > 0 else None
            record["E_wmax"] = err / w_max if np.isfinite(w_max) and w_max > 0 else None
        records.append(record)
    return pd.DataFrame(records)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(finite.max()),
    }


def grouped_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    """Summary over all crops, per group, and per group x tree."""

    def block(frame: pd.DataFrame) -> dict[str, Any]:
        with_pred = frame[frame["has_prediction"]]
        out: dict[str, Any] = {
            "n_crops": int(len(frame)),
            "n_with_prediction": int(len(with_pred)),
            "n_reject_no_prediction": int((frame["has_prediction"] == False).sum()),  # noqa: E712
        }
        for key in METRIC_KEYS:
            out[key] = _quantiles(with_pred[key].to_numpy(dtype=np.float64))
        return out

    result: dict[str, Any] = {
        "schema_version": POSITION_METRICS_SCHEMA,
        "overall": block(metrics),
        "by_group": {g: block(metrics[metrics["group"] == g]) for g in GROUP_ORDER},
        "by_group_tree": {
            g: {
                tree: block(metrics[(metrics["group"] == g) & (metrics["tree_id"] == tree)])
                for tree in sorted(metrics[metrics["group"] == g]["tree_id"].unique())
            }
            for g in GROUP_ORDER
        },
        "E_mm_status": {
            "available": int(metrics["mm_per_pixel"].notna().sum()),
            "unavailable": int(metrics["mm_per_pixel"].isna().sum()),
            "policy": "report empty; never guess from tree diameter, camera model or DPI tags",
        },
    }
    return result
