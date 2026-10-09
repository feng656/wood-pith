#!/usr/bin/env python3
"""Stage 6: parent/fragment screening and exact parent delete-refit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stage3_all_arc import (certificate, finite_h, objective_components,
                            pack_rings, phase_mesh, point_from_h, relevant,
                            rp_distance, run_level, select_starts)
from stage4_profiles import component_summary, components


HESSIAN_STEP = 1.0e-3
RIDGE = 1.0e-6
EXACT_ALL_MAX_RINGS = 8
PROFILE_PHI_BINS = 18
PROFILE_THETA_BINS = 12
PROFILE_DELTA = 0.75
GT_EQUIV_RAD = 0.005
HIGH_SHIFT_RAD = 0.02


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


def numeric_hessian(fun, p: np.ndarray) -> np.ndarray:
    e = HESSIAN_STEP
    f0 = float(fun(p)); h = np.empty((2, 2), dtype=float)
    for i in range(2):
        ei = np.zeros(2); ei[i] = e
        h[i, i] = (float(fun(p + ei)) - 2.0 * f0 + float(fun(p - ei))) / (e * e)
    e0 = np.array([e, 0.0]); e1 = np.array([0.0, e])
    mixed = (float(fun(p + e0 + e1)) - float(fun(p + e0 - e1)) -
             float(fun(p - e0 + e1)) + float(fun(p - e0 - e1))) / (4.0 * e * e)
    h[0, 1] = h[1, 0] = mixed
    return h


def numeric_gradient(fun, p: np.ndarray) -> np.ndarray:
    e = HESSIAN_STEP
    return np.array([(float(fun(p + np.eye(2)[i] * e)) -
                      float(fun(p - np.eye(2)[i] * e))) / (2.0 * e) for i in range(2)])


def screen_parents(h: list[float], rings: dict[str, list[dict[str, Any]]]):
    p = point_from_h(h)
    full = pack_rings(rings)
    if p is None:
        return [], {"valid": False, "reason": "INFINITY_TOUCHING"}
    fun = lambda z: objective_components(finite_h(z), full)["loss"]
    hessian = numeric_hessian(fun, p)
    eig = np.linalg.eigvalsh(hessian)
    condition = float(np.linalg.cond(hessian)) if np.all(np.isfinite(hessian)) else float("inf")
    inverse = np.linalg.pinv(hessian + RIDGE * np.eye(2))
    rows = []
    for ring_id, arcs in rings.items():
        group = pack_rings({ring_id: arcs})
        group_fun = lambda z: objective_components(finite_h(z), group)["J_abs"] / max(len(rings), 1)
        gradient = numeric_gradient(group_fun, p)
        shift = inverse @ gradient
        group_loss = float(objective_components(h, group)["J_abs"])
        rows.append({"ring_id": ring_id, "if_shift_norm": float(np.linalg.norm(shift)),
                     "if_shift_vector": shift.tolist(), "group_loss_at_full": group_loss,
                     "arc_count": len(arcs),
                     "support_length_norm": float(sum(a["length_norm"] for a in arcs)),
                     "influence_valid": bool(eig[0] > 1.0e-6 and condition < 1.0e8),
                     "exact_status": "SCREEN_ONLY"})
    losses = np.array([r["group_loss_at_full"] for r in rows])
    median = float(np.median(losses)) if len(losses) else 0.0
    for row in rows:
        row["screen_conflict"] = float(row["group_loss_at_full"] - median)
    return rows, {"valid": True, "hessian": hessian.tolist(),
                  "eigenvalues": eig.tolist(), "condition": condition}


def choose_exact(screening: list[dict[str, Any]], state: str, crop_id: str) -> set[str]:
    if state not in {"POINT_CANDIDATE", "RANGE_UNCERTAIN"}:
        return set()
    if len(screening) <= EXACT_ALL_MAX_RINGS:
        return {str(row["ring_id"]) for row in screening}
    influence = sorted(screening, key=lambda row: row["if_shift_norm"], reverse=True)[:2]
    conflict = max(screening, key=lambda row: row["screen_conflict"])
    digest = int(hashlib.sha256(crop_id.encode("utf-8")).hexdigest()[:8], 16)
    control = screening[digest % len(screening)]
    return {str(row["ring_id"]) for row in influence + [conflict, control]}


def coarse_profile(evidence: dict[str, Any], j_min: float) -> dict[str, Any]:
    phis = np.linspace(-math.pi, math.pi, PROFILE_PHI_BINS, endpoint=False)
    thetas = np.linspace(0.0, math.atan(50.0), PROFILE_THETA_BINS)
    loss = np.empty((len(thetas), len(phis)), dtype=float)
    from stage3_all_arc import far_h
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            loss[i, j] = float(objective_components(far_h(float(phi), math.tan(float(theta))), evidence)["loss"])
    mask = loss <= j_min + PROFILE_DELTA
    comps = components(mask)
    return {"resolution": [PROFILE_THETA_BINS, PROFILE_PHI_BINS],
            "delta": PROFILE_DELTA, "component_count": len(comps),
            "components": [component_summary(c, phis, thetas) for c in comps]}


def simple_state(profile: dict[str, Any]) -> str:
    if profile["component_count"] == 0:
        return "REJECT"
    if profile["component_count"] > 1:
        return "MULTIMODAL"
    comp = profile["components"][0]
    if comp["touches_infinity"]:
        return "AXIS"
    if comp["W_phi_rad"] > math.radians(45.0):
        return "REJECT"
    if comp["W_phi_rad"] > math.radians(15.0) or comp["W_log_r"] is None or comp["W_log_r"] > math.log(2.0):
        return "RANGE_UNCERTAIN"
    return "POINT_CANDIDATE"


def exact_delete(full_h: list[float], remaining: dict[str, list[dict[str, Any]]], phase):
    if len(remaining) < 2:
        return None, {"passed": False, "reason_codes": ["TOO_FEW_REMAINING_PARENTS"]}, None
    evidence = pack_rings(remaining)
    full_start = {"h": full_h, "source": "FULL_SOLUTION_WARM"}
    phase_start = select_starts(phase, evidence, 1, 1)
    basins_b = run_level([full_start], evidence, 1, 45)
    basins_2b = run_level([full_start] + phase_start, evidence, 1, 90)
    phase_basins = run_level(phase_start, evidence, 1, 90)
    cert = certificate(basins_b, basins_2b, phase_basins, evidence)
    rel = relevant(basins_2b)
    best = rel[0] if rel else None
    profile = coarse_profile(evidence, float(best["J_abs"])) if best and cert["passed"] else None
    return best, cert, profile


def roles(full_state: str, deleted_state: str, shift: float, gt_help: float | None,
          c_phi: float | None, c_range: float | None, mode_added: int) -> list[str]:
    output = []
    if deleted_state in {"AXIS", "MULTIMODAL", "REJECT"} and full_state not in {"AXIS", "MULTIMODAL", "REJECT"}:
        output.append("DIRECTION_CRITICAL")
    elif c_phi is not None and c_phi > 0.2:
        output.append("DIRECTION_CRITICAL")
    if full_state == "POINT_CANDIDATE" and deleted_state in {"RANGE_UNCERTAIN", "AXIS"}:
        output.append("RANGE_CRITICAL")
    elif c_range is not None and c_range > 0.2:
        output.append("RANGE_CRITICAL")
    if mode_added > 0:
        output.append("MODE_EXCLUSION")
    if gt_help is not None and gt_help > GT_EQUIV_RAD:
        output.append("BENEFICIAL_GT")
    elif gt_help is not None and gt_help < -GT_EQUIV_RAD:
        output.append("HARMFUL_GT")
    if shift > HIGH_SHIFT_RAD:
        output.append("HIGH_LEVERAGE")
    if not output and shift <= GT_EQUIV_RAD and (gt_help is None or abs(gt_help) <= GT_EQUIV_RAD):
        output.append("REDUNDANT")
    return output or ["UNRESOLVED"]


def nominal_component(state_row: dict[str, Any]) -> dict[str, Any] | None:
    support = state_row.get("nominal_support", {})
    comps = support.get("components", [])
    return comps[0] if len(comps) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--stage4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    crops = iter_jsonl(args.stage1_dir / "crop_summary.jsonl")
    states = iter_jsonl(args.stage4_dir / "states.jsonl")
    fragments = iter_jsonl(args.stage1_dir / "fragments.jsonl")
    pending = next(fragments, None); phase = phase_mesh()
    screening_rows, exact_rows, fragment_rows, crop_rows = [], [], [], []

    for index, (crop, state_row) in enumerate(zip(crops, states), 1):
        if args.limit > 0 and index > args.limit:
            break
        crop_id = str(crop["crop_id"])
        if str(state_row["crop_id"]) != crop_id:
            raise RuntimeError(f"stage input order mismatch at {crop_id}")
        arcs = []
        while pending is not None and str(pending["crop_id"]) == crop_id:
            arcs.append(pending); pending = next(fragments, None)
        rings: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for arc in arcs:
            rings[str(arc["ring_id"])].append(arc)
            fragment_rows.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                                  "ring_id": str(arc["ring_id"]), "fragment_id": arc["fragment_id"],
                                  "support_length_norm": arc["length_norm"],
                                  "theta_total_rad": arc["theta_total_rad"],
                                  "uncertainty_source": arc["uncertainty_source"],
                                  "exact_status": "SCREEN_ONLY_PARENT_FIRST"})
        if state_row.get("h") is None:
            crop_rows.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                              "exact_count": 0, "reason": "NO_VALID_FULL_SOLUTION"})
            continue
        screens, hessian_info = screen_parents(state_row["h"], rings)
        chosen = choose_exact(screens, str(state_row["state"]), crop_id)
        screen_lookup = {str(row["ring_id"]): row for row in screens}
        for row in screens:
            row.update({"sample_id": crop["sample_id"], "crop_id": crop_id})
            if str(row["ring_id"]) in chosen:
                row["exact_status"] = "SELECTED_FOR_EXACT"
            screening_rows.append(row)
        full_comp = nominal_component(state_row)
        pith = crop.get("pith_px")
        h_gt = None
        if pith is not None:
            p_norm = (np.asarray(pith, dtype=float) - np.asarray(crop["center_px"], dtype=float)) / float(crop["scale_px"])
            h_gt = finite_h(p_norm)
        exact_ok = 0
        for ring_id in sorted(chosen, key=lambda value: (len(value), value)):
            remaining = {rid: aa for rid, aa in rings.items() if rid != ring_id}
            best, cert, profile = exact_delete(state_row["h"], remaining, phase)
            if best is None or not cert["passed"] or profile is None:
                exact_rows.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                                   "ring_id": ring_id, "status": "UNRESOLVED_SEARCH",
                                   "reason_codes": cert.get("reason_codes", [])})
                continue
            exact_ok += 1
            deleted_h = best["h"]; deleted_state = simple_state(profile)
            shift = rp_distance(state_row["h"], deleted_h)
            gt_help = (rp_distance(deleted_h, h_gt) - rp_distance(state_row["h"], h_gt)) if h_gt is not None else None
            delete_comp = profile["components"][0] if profile["component_count"] == 1 else None
            c_phi = None; c_range = None
            if full_comp and delete_comp:
                c_phi = math.log((float(delete_comp["W_phi_rad"]) + 1.0e-6) /
                                 (float(full_comp["W_phi_rad"]) + 1.0e-6))
                if full_comp.get("W_log_r") is not None and delete_comp.get("W_log_r") is not None:
                    c_range = math.log((float(delete_comp["W_log_r"]) + 1.0e-6) /
                                       (float(full_comp["W_log_r"]) + 1.0e-6))
            mode_added = max(0, int(profile["component_count"]) - int(state_row.get("nominal_support", {}).get("component_count", 1)))
            remaining_evidence = pack_rings(remaining)
            at_full = objective_components(state_row["h"], remaining_evidence)
            at_delete = objective_components(deleted_h, remaining_evidence)
            label = roles(str(state_row["state"]), deleted_state, shift, gt_help, c_phi, c_range, mode_added)
            exact_rows.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                               "ring_id": ring_id, "status": "EXACT",
                               "full_state": state_row["state"], "deleted_state": deleted_state,
                               "deleted_h": deleted_h, "search_certificate": cert,
                               "C_shift_proj": shift, "C_GT_proj": gt_help,
                               "C_GT_mm": None, "C_GT_mm_status": "PIXEL_ONLY_NO_MM_PER_PIXEL",
                               "C_phi": c_phi, "C_range": c_range,
                               "C_mode_components_added": mode_added,
                               "C_conflict": float(at_full["J_abs"] - at_delete["J_abs"]),
                               "C50": float(at_full["M50"] - at_delete["M50"]),
                               "C_tail": float(at_full["U20"] - at_delete["U20"]),
                               "roles": label, "deleted_profile": profile})
        crop_rows.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                          "parent_count": len(rings), "selected_exact_count": len(chosen),
                          "resolved_exact_count": exact_ok, "hessian_screen": hessian_info})

    write_jsonl(args.output_dir / "parent_screening.jsonl", screening_rows)
    write_jsonl(args.output_dir / "parent_exact_contributions.jsonl", exact_rows)
    write_jsonl(args.output_dir / "fragment_screening.jsonl", fragment_rows)
    write_jsonl(args.output_dir / "crop_contribution_summary.jsonl", crop_rows)
    exact = [row for row in exact_rows if row["status"] == "EXACT"]
    summary = {"stage": 6, "schema": "ArcPith-GT-v4-stage6",
               "crops": len(crop_rows), "parent_screening_records": len(screening_rows),
               "fragment_screening_records": len(fragment_rows),
               "exact_selected": len(exact_rows), "exact_resolved": len(exact),
               "exact_unresolved": len(exact_rows) - len(exact),
               "role_counts": dict(Counter(role for row in exact for role in row["roles"])),
               "physical_gt_contribution": "NOT_AVAILABLE_PIXEL_ONLY",
               "exact_policy": {"all_parents_when_R_le": EXACT_ALL_MAX_RINGS,
                                "large_R": "top2 influence + top conflict + deterministic control"}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "Stage 6 screens every parent and fragment, then performs certified exact parent delete-refit "
        "under the registered audit budget. Only resolved exact groups receive formal roles. "
        "Projective GT contribution is available; millimetre contribution is withheld because scale is absent.\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if crop_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
