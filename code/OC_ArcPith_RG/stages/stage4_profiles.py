#!/usr/bin/env python3
"""Stage 4: high-resolution All-Arc refinement and projective profiles."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from stage3_all_arc import (FINITE_BOUND, TAU_H, canon, far_h, finite_h,
                            objective_components, optimize, pack_rings,
                            phi_kappa, point_from_h, rp_distance)


DELTA_FAMILY = (0.25, 0.50, 0.75)
PHI_BINS = 36
THETA_BINS = 24
KAPPA_MAX = 50.0
NUMERICAL_SHIFT_TOL = 0.01
POINT_PHI_WIDTH = math.radians(15.0)
RANGE_PHI_WIDTH = math.radians(45.0)
POINT_LOG_RANGE_WIDTH = math.log(2.0)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")


def densify_arcs(arcs: list[dict[str, Any]], factor: int = 2) -> list[dict[str, Any]]:
    """Interpolate frozen Stage-1 splines to denser quadrature nodes."""
    out: list[dict[str, Any]] = []
    for arc in arcs:
        x = np.asarray(arc["points_norm"], dtype=float)
        t = np.asarray(arc["tangents"], dtype=float)
        if len(x) < 2:
            continue
        old = np.linspace(0.0, 1.0, len(x))
        new = np.linspace(0.0, 1.0, factor * (len(x) - 1) + 1)
        xx = np.column_stack([np.interp(new, old, x[:, k]) for k in range(2)])
        tt = np.column_stack([np.interp(new, old, t[:, k]) for k in range(2)])
        tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1.0e-15)
        segment = np.linalg.norm(np.diff(xx, axis=0), axis=1)
        ds = np.empty(len(xx), dtype=float)
        ds[0], ds[-1] = segment[0] / 2.0, segment[-1] / 2.0
        ds[1:-1] = (segment[:-1] + segment[1:]) / 2.0
        dense = dict(arc)
        dense["points_norm"] = xx.tolist()
        dense["tangents"] = tt.tolist()
        dense["ds_norm"] = ds.tolist()
        dense["quadrature_high_resolution_nodes"] = len(xx)
        out.append(dense)
    return out


def circular_width(mask: np.ndarray, period: float) -> float:
    idx = np.flatnonzero(mask)
    if len(idx) <= 1:
        return 0.0
    angles = idx.astype(float) * period / len(mask)
    gaps = np.diff(np.r_[angles, angles[0] + period])
    return float(period - np.max(gaps))


def components(mask: np.ndarray) -> list[np.ndarray]:
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found: list[np.ndarray] = []
    for start in np.argwhere(mask):
        i, j = map(int, start)
        if seen[i, j]:
            continue
        cells: list[tuple[int, int]] = []
        queue = deque([(i, j)]); seen[i, j] = True
        while queue:
            r, c = queue.popleft(); cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, (c - 1) % cols), (r, (c + 1) % cols)):
                if 0 <= rr < rows and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True; queue.append((rr, cc))
        component = np.zeros_like(mask, dtype=bool)
        for cell in cells:
            component[cell] = True
        found.append(component)
    found.sort(key=lambda item: int(np.sum(item)), reverse=True)
    return found


def grid_profile(evidence: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phis = np.linspace(-math.pi, math.pi, PHI_BINS, endpoint=False)
    thetas = np.linspace(0.0, math.atan(KAPPA_MAX), THETA_BINS)
    loss = np.empty((THETA_BINS, PHI_BINS), dtype=float)
    for i, theta in enumerate(thetas):
        kappa = math.tan(float(theta))
        for j, phi in enumerate(phis):
            loss[i, j] = float(objective_components(far_h(float(phi), kappa), evidence)["loss"])
    return phis, thetas, loss


def component_summary(component: np.ndarray, phis: np.ndarray, thetas: np.ndarray) -> dict[str, Any]:
    rows, cols = np.where(component)
    touches_axis = bool(np.any(rows == 0))
    phi_mask = np.any(component, axis=0)
    phi_period = math.pi if touches_axis else 2.0 * math.pi
    # For an infinity-touching component, collapse antipodal phi bins.
    if touches_axis:
        half = len(phi_mask) // 2
        phi_mask = phi_mask[:half] | phi_mask[half:half * 2]
    width_phi = circular_width(phi_mask, phi_period)
    theta_min, theta_max = float(np.min(thetas[rows])), float(np.max(thetas[rows]))
    kappa_min, kappa_max = math.tan(theta_min), math.tan(theta_max)
    if kappa_min <= 1.0e-12:
        interval = [1.0 / max(kappa_max, 1.0e-12), None]
        log_width = None
    else:
        r_min, r_max = 1.0 / max(kappa_max, 1.0e-12), 1.0 / kappa_min
        interval = [r_min, r_max]
        log_width = float(math.log(max(r_max / max(r_min, 1.0e-15), 1.0)))
    return {"cell_count": int(np.sum(component)), "touches_infinity": touches_axis,
            "W_phi_rad": width_phi, "W_vartheta_rad": theta_max - theta_min,
            "vartheta_interval_rad": [theta_min, theta_max],
            "kappa_interval": [kappa_min, kappa_max],
            "range_over_fov_interval": interval, "W_log_r": log_width}


def support_family(loss: np.ndarray, phis: np.ndarray, thetas: np.ndarray,
                   j_min: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for delta in DELTA_FAMILY:
        mask = loss <= j_min + delta
        comps = components(mask)
        output.append({"delta": delta, "support_cells": int(np.sum(mask)),
                       "component_count": len(comps),
                       "components": [component_summary(c, phis, thetas) for c in comps]})
    return output


def classify(search_passed: bool, numerical_stable: bool, retained_modes: int,
             family: list[dict[str, Any]]) -> tuple[str, list[str]]:
    if not search_passed:
        return "REJECT", ["SEARCH_CERTIFICATE_FAILED"]
    if not numerical_stable:
        return "REJECT", ["NUMERICAL_UNSTABLE"]
    nominal = family[-1]
    counts = [item["component_count"] for item in family]
    if not counts or nominal["component_count"] == 0:
        return "REJECT", ["EMPTY_SUPPORT"]
    if max(counts) - min(counts) > 1:
        return "REJECT", ["TOPOLOGY_FRAGILE"]
    if retained_modes >= 2 or nominal["component_count"] >= 2:
        return "MULTIMODAL", ["MULTIPLE_PERSISTENT_COMPONENTS"]
    comp = nominal["components"][0]
    if comp["touches_infinity"]:
        if comp["W_phi_rad"] <= RANGE_PHI_WIDTH:
            return "AXIS", ["SUPPORT_TOUCHES_INFINITY"]
        return "REJECT", ["DIRECTION_NOT_IDENTIFIABLE"]
    if comp["W_phi_rad"] > RANGE_PHI_WIDTH:
        return "REJECT", ["DIRECTION_NOT_IDENTIFIABLE"]
    if comp["W_phi_rad"] > POINT_PHI_WIDTH or comp["W_log_r"] is None or comp["W_log_r"] > POINT_LOG_RANGE_WIDTH:
        return "RANGE_UNCERTAIN", ["FINITE_RANGE_SUPPORT_WIDE"]
    return "POINT_CANDIDATE", ["UNCALIBRATED_FOUR_TREE_DATASET"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--stage3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    crops = iter_jsonl(args.stage1_dir / "crop_summary.jsonl")
    estimates = iter_jsonl(args.stage3_dir / "all_arc_estimates.jsonl")
    fragments = iter_jsonl(args.stage1_dir / "fragments.jsonl")
    pending = next(fragments, None)
    states: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    refinements: list[dict[str, Any]] = []

    for index, (crop, estimate) in enumerate(zip(crops, estimates), 1):
        if args.limit > 0 and index > args.limit:
            break
        crop_id = str(crop["crop_id"])
        if str(estimate["crop_id"]) != crop_id:
            raise RuntimeError(f"stage input order mismatch at {crop_id}")
        arcs: list[dict[str, Any]] = []
        while pending is not None and str(pending["crop_id"]) == crop_id:
            arcs.append(pending); pending = next(fragments, None)

        if not estimate.get("search_certificate_passed") or estimate.get("h") is None:
            states.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                           "state": "REJECT", "reason_codes": ["SEARCH_CERTIFICATE_FAILED"],
                           "pith_px": None, "calibrated_coverage": False})
            continue
        dense_arcs = densify_arcs(arcs)
        ring_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for arc in dense_arcs:
            ring_map[str(arc["ring_id"])].append(arc)
        evidence = pack_rings(ring_map)
        h0 = canon(estimate["h"])
        p0 = point_from_h(h0)
        chart = "finite" if p0 is not None and float(np.max(np.abs(p0))) < FINITE_BOUND else "far"
        refined = optimize(h0, evidence, chart, 1, 180)
        if refined is None:
            states.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                           "state": "REJECT", "reason_codes": ["REFINEMENT_FAILED"],
                           "pith_px": None, "calibrated_coverage": False})
            continue
        shift = rp_distance(h0, refined["h"])
        numerical_stable = bool(shift <= NUMERICAL_SHIFT_TOL and refined["stationary"])
        point = point_from_h(refined["h"])
        point_px = (np.asarray(crop["center_px"]) + float(crop["scale_px"]) * point).tolist() if point is not None else None
        phis, thetas, loss = grid_profile(evidence)
        j_min = min(float(refined["J_abs"]), float(np.min(loss)))
        family = support_family(loss, phis, thetas, j_min)
        state, reasons = classify(True, numerical_stable,
                                  int(estimate.get("retained_mode_count", 1)), family)
        state_row = {"sample_id": crop["sample_id"], "tree_id": crop["tree_id"],
                     "section_id": crop["section_id"], "crop_id": crop_id,
                     "state": state, "reason_codes": reasons,
                     "h": refined["h"], "pith_norm": point.tolist() if point is not None else None,
                     "pith_px": point_px, "J_abs": float(refined["J_abs"]),
                     "M50": float(refined["M50"]), "U20": float(refined["U20"]),
                     "calibrated_coverage": False,
                     "support_threshold_status": "DELTA_FAMILY_ONLY_INSUFFICIENT_INDEPENDENT_TREES",
                     "nominal_support": family[-1]}
        states.append(state_row)
        refinements.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                            "input_h": h0.tolist(), "refined_h": refined["h"],
                            "rp2_shift_rad": shift, "numerical_stable": numerical_stable,
                            "stationary": refined["stationary"], "gradient_norm": refined["gradient_norm"],
                            "coarse_J_abs": estimate["J_abs"], "refined_J_abs": refined["J_abs"],
                            "quadrature_factor": 2})
        profiles.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                         "phi_rad": np.round(phis, 6).tolist(),
                         "vartheta_rad": np.round(thetas, 6).tolist(),
                         "L_phi": np.round(np.min(loss, axis=0), 6).tolist(),
                         "L_vartheta": np.round(np.min(loss, axis=1), 6).tolist(),
                         "grid_shape": list(loss.shape), "J_min": j_min,
                         "delta_family": family})

    write_jsonl(args.output_dir / "states.jsonl", states)
    write_jsonl(args.output_dir / "profiles.jsonl", profiles)
    write_jsonl(args.output_dir / "refinements.jsonl", refinements)
    summary = {"stage": 4, "schema": "ArcPith-GT-v4-stage4", "records": len(states),
               "profiled_records": len(profiles),
               "state_counts": dict(Counter(row["state"] for row in states)),
               "numerical_unstable": sum(not row["numerical_stable"] for row in refinements),
               "profile_grid": {"phi_bins": PHI_BINS, "vartheta_bins": THETA_BINS,
                                "kappa_max": KAPPA_MAX},
               "delta_family": list(DELTA_FAMILY), "calibrated_coverage": False,
               "calibration_note": "Only four biological trees are available; POINT is forbidden at Stage 4."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "Stage 4 refines certified All-Arc basins with doubled quadrature and emits direction, "
        "projective-distance, finite-range and nested support summaries. With only four independent "
        "trees, thresholds are not coverage-calibrated and no record is promoted to POINT.\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if states else 2


if __name__ == "__main__":
    raise SystemExit(main())
