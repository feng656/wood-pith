#!/usr/bin/env python3
"""Stage 5: model-risk diagnostics and non-scalar ExecutionDegree."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from stage3_all_arc import EPS_V, SIGMA0, SIGMA_MAX, SIGMA_MIN, canon


STRUCTURED_R2_BLOCK = 0.25


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


def standardized_residuals(h: list[float], arcs: list[dict[str, Any]]):
    hh = canon(h); a, h0 = hh[:2], float(hh[2])
    z_all, theta_all, parent_all = [], [], []
    if h0 <= 1.0e-10:
        return np.empty(0), np.empty(0), np.empty(0, dtype=str)
    p = a / h0
    for arc in arcs:
        x = np.asarray(arc["points_norm"], dtype=float)
        t = np.asarray(arc["tangents"], dtype=float)
        v = a[None, :] - h0 * x
        d = np.sqrt(np.sum(v * v, axis=1) + EPS_V * EPS_V)
        numer = np.sum(t * v, axis=1)
        e = numer / d
        grad_v = t / d[:, None] - numer[:, None] * v / (d ** 3)[:, None]
        jx = abs(h0) * np.linalg.norm(grad_v, axis=1)
        t_perp = np.column_stack([-t[:, 1], t[:, 0]])
        jpsi = np.sum(t_perp * v, axis=1) / d
        sigma = np.sqrt((jx * float(arc.get("sigma_position_norm", 0.0))) ** 2 +
                        (jpsi * float(arc.get("sigma_psi_rad", 0.0))) ** 2 + SIGMA0 ** 2)
        sigma = np.clip(sigma, SIGMA_MIN, SIGMA_MAX)
        z_all.append(e / sigma)
        delta = x - p[None, :]
        theta_all.append(np.arctan2(delta[:, 1], delta[:, 0]))
        parent_all.append(np.full(len(x), str(arc["ring_id"]), dtype=object))
    return np.concatenate(z_all), np.concatenate(theta_all), np.concatenate(parent_all)


def structured_oof(h: list[float] | None, arcs: list[dict[str, Any]]) -> dict[str, Any]:
    if h is None:
        return {"applicable": False, "reason": "NO_FINITE_ESTIMATE"}
    z, theta, parent = standardized_residuals(h, arcs)
    groups = np.unique(parent)
    if len(z) < 20 or len(groups) < 3:
        return {"applicable": False, "reason": "INSUFFICIENT_PARENT_SUPPORT"}
    X = np.column_stack([np.ones(len(theta)), np.sin(theta), np.cos(theta),
                         np.sin(2.0 * theta), np.cos(2.0 * theta)])
    model_sse = 0.0; null_sse = 0.0; folds = 0
    for group in groups:
        test = parent == group; train = ~test
        if int(np.sum(test)) < 2 or int(np.sum(train)) < X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X[train], z[train], rcond=None)
        prediction = X[test] @ beta
        null = float(np.mean(z[train]))
        model_sse += float(np.sum((z[test] - prediction) ** 2))
        null_sse += float(np.sum((z[test] - null) ** 2))
        folds += 1
    if folds < 3 or null_sse <= 1.0e-12:
        return {"applicable": False, "reason": "OOF_DEGENERATE"}
    r2 = float(1.0 - model_sse / null_sse)
    return {"applicable": True, "basis": "1,sin(theta),cos(theta),sin(2theta),cos(2theta)",
            "cross_fit": "leave-one-parent-ring-out", "folds": folds,
            "R2_struct_oof": r2, "model_sse": model_sse, "null_sse": null_sse,
            "systematic_structure_flag": r2 >= STRUCTURED_R2_BLOCK}


def risk_and_degree(state: str, diagnostic: dict[str, Any], target_domain: str,
                    search_passed: bool) -> tuple[str, str, list[str]]:
    if state == "REJECT" or not search_passed:
        return "HARD_REJECT", "X0_INVALID", ["GEOMETRY_OR_SEARCH_INVALID"]
    if diagnostic.get("systematic_structure_flag"):
        risk = "POINT_BLOCKING"; reasons = ["STRUCTURED_RESIDUAL_OOF_HIGH"]
    else:
        risk = "UNKNOWN"; reasons = ["TARGET_ALIGNMENT_UNKNOWN", "CALIBRATION_POWER_INSUFFICIENT"]
    if state == "MULTIMODAL":
        return risk, "X1_GEOMETRY_WEAK", reasons + ["MULTIMODAL_GEOMETRY"]
    if state in {"AXIS", "RANGE_UNCERTAIN"}:
        return risk, "X2_DIRECTION_USABLE", reasons
    if state == "POINT_CANDIDATE":
        return risk, "X3_FINITE_RESEARCH", reasons
    return "HARD_REJECT", "X0_INVALID", ["UNKNOWN_GEOMETRY_STATE"]


def execution_profile(state: str, risk: str, search_passed: bool,
                      structured: dict[str, Any]) -> dict[str, str]:
    direction = ("COMPACT" if state == "POINT_CANDIDATE" else
                 "USABLE" if state in {"AXIS", "RANGE_UNCERTAIN"} else "WEAK")
    range_status = ("FINITE_COMPACT" if state == "POINT_CANDIDATE" else
                    "FINITE_WIDE" if state == "RANGE_UNCERTAIN" else
                    "INFINITY_TOUCHING" if state == "AXIS" else "UNUSABLE")
    return {"E_data": "PIXEL_ONLY_LINEAGE_VALIDATED",
            "E_search": "CERTIFIED" if search_passed else "FAILED",
            "E_direction": direction, "E_range": range_status,
            "E_model": risk,
            "E_robust": "STRUCTURED_RESIDUAL_CHECKED_DELETION_PENDING_STAGE6" if structured.get("applicable") else "INCOMPLETE",
            "E_cal": "UNCalibrated_INSUFFICIENT_INDEPENDENT_TREES"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-dir", type=Path, required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--stage3-dir", type=Path, required=True)
    parser.add_argument("--stage4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = iter_jsonl(args.stage0_dir / "manifest_validated.jsonl")
    crops = iter_jsonl(args.stage1_dir / "crop_summary.jsonl")
    estimates = iter_jsonl(args.stage3_dir / "all_arc_estimates.jsonl")
    states = iter_jsonl(args.stage4_dir / "states.jsonl")
    fragments = iter_jsonl(args.stage1_dir / "fragments.jsonl")
    pending = next(fragments, None)
    output: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for index, (source, crop, estimate, state_row) in enumerate(zip(manifest, crops, estimates, states), 1):
        if args.limit > 0 and index > args.limit:
            break
        crop_id = str(crop["crop_id"])
        if any(str(row["crop_id"]) != crop_id for row in (source, estimate, state_row)):
            raise RuntimeError(f"stage input order mismatch at {crop_id}")
        arcs: list[dict[str, Any]] = []
        while pending is not None and str(pending["crop_id"]) == crop_id:
            arcs.append(pending); pending = next(fragments, None)
        state = str(state_row["state"])
        structured = structured_oof(state_row.get("h") if state != "REJECT" else None, arcs)
        target_domain = str(source.get("target_domain", source.get("metadata", {}).get("target_domain", "TARGET_UNKNOWN")))
        search_passed = bool(estimate.get("search_certificate_passed"))
        risk, degree, reasons = risk_and_degree(state, structured, target_domain, search_passed)
        profile = execution_profile(state, risk, search_passed, structured)
        output.append({"sample_id": crop["sample_id"], "tree_id": crop["tree_id"],
                       "section_id": crop["section_id"], "crop_id": crop_id,
                       "geometry_state": state, "risk_status": risk,
                       "execution_degree": degree, "reason_codes": reasons,
                       "execution_profile": profile,
                       "h": state_row.get("h"), "pith_px": state_row.get("pith_px"),
                       "coordinate_source": "STAGE4_ALL_ARC_UNCHANGED",
                       "target_domain": target_domain, "calibrated_probability": None})
        diagnostics.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                            "structured_residual": structured,
                            "target_alignment": {"status": target_domain,
                                                 "registry_available": False},
                            "parent_delete_scatter": {"status": "PENDING_STAGE6_EXACT_DELETE_REFIT"},
                            "measurement_search_replay": {"stage3_certificate_passed": search_passed,
                                                          "stage4_numerical_stable": "NUMERICAL_UNSTABLE" not in state_row.get("reason_codes", [])},
                            "common_bias_fixtures": {"status": "NOT_CALIBRATED_FOUR_TREE_DATASET"}})

    write_jsonl(args.output_dir / "execution.jsonl", output)
    write_jsonl(args.output_dir / "risk_diagnostics.jsonl", diagnostics)
    summary = {"stage": 5, "schema": "ArcPith-GT-v4-stage5", "records": len(output),
               "risk_counts": dict(Counter(row["risk_status"] for row in output)),
               "execution_degree_counts": dict(Counter(row["execution_degree"] for row in output)),
               "structured_residual_applicable": sum(row["structured_residual"].get("applicable", False) for row in diagnostics),
               "structured_residual_blocking": sum(row["structured_residual"].get("systematic_structure_flag", False) for row in diagnostics),
               "coordinate_changed": False, "maximum_degree_allowed": "X3_FINITE_RESEARCH",
               "calibrated_probability_emitted": False}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "Stage 5 keeps the Stage-4 All-Arc coordinate unchanged, cross-fits a low-capacity "
        "structured-residual diagnostic, and writes categorical ExecutionProfile/ExecutionDegree. "
        "Unknown target alignment and four-tree power prevent LOW_RISK, X4, X5, or calibrated probabilities.\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if output else 2


if __name__ == "__main__":
    raise SystemExit(main())
