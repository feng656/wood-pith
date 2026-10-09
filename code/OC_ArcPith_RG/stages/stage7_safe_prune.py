#!/usr/bin/env python3
"""Stage 7: enforce Safe-Prune deployment gates and preserve All-Arc."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MIN_FREEZE_TREES = 30


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-dir", type=Path, required=True)
    parser.add_argument("--stage5-dir", type=Path, required=True)
    parser.add_argument("--stage6-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    states = read_jsonl(args.stage4_dir / "states.jsonl")
    execution = read_jsonl(args.stage5_dir / "execution.jsonl")
    exact = read_jsonl(args.stage6_dir / "parent_exact_contributions.jsonl")
    if len(states) != len(execution):
        raise RuntimeError("Stage 4 and Stage 5 record counts differ")
    trees = sorted({str(row["tree_id"]) for row in execution})
    gate_reasons = [
        f"INDEPENDENT_TREE_COUNT_{len(trees)}_BELOW_{MIN_FREEZE_TREES}",
        "NO_POWER_BASED_N_FREEZE_ESTIMATE",
        "NO_INDEPENDENT_SAFE_PRUNE_VALIDATION",
        "FALSE_POINT_NONINFERIORITY_NOT_ESTABLISHED",
        "CRITICAL_ARC_FALSE_REMOVAL_RATE_NOT_ESTABLISHED",
        "TARGET_ALIGNMENT_UNKNOWN",
        "MEASUREMENT_GRID_REPLAY_NOT_FULLY_CALIBRATED",
    ]
    deployment_gate = {
        "safe_prune_enabled": False, "decision": "OFF",
        "independent_tree_count": len(trees), "tree_ids": trees,
        "minimum_tree_floor": MIN_FREEZE_TREES, "n_power": None,
        "passed": False, "reason_codes": gate_reasons,
        "fallback": "F-All high-resolution All-Arc",
        "oracle_may_change_final_coordinate": False,
    }

    by_crop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exact:
        if row.get("status") == "EXACT":
            by_crop[str(row["crop_id"])].append(row)

    final_rows, oracle_rows, suspect_rows = [], [], []
    for state, exec_row in zip(states, execution):
        crop_id = str(state["crop_id"])
        if str(exec_row["crop_id"]) != crop_id:
            raise RuntimeError(f"stage input order mismatch at {crop_id}")
        final_rows.append({
            "sample_id": state["sample_id"], "tree_id": exec_row["tree_id"],
            "section_id": exec_row["section_id"], "crop_id": crop_id,
            "final_method": "F-All", "safe_prune_applied": False,
            "safe_prune_status": "OFF_DATASET_DEPLOYMENT_GATE_FAILED",
            "h": state.get("h"), "pith_px": state.get("pith_px"),
            "geometry_state": state["state"], "risk_status": exec_row["risk_status"],
            "execution_degree": exec_row["execution_degree"],
            "coordinate_source": "STAGE4_HIGH_RESOLUTION_ALL_ARC",
            "reason_codes": gate_reasons,
        })
        candidates = by_crop.get(crop_id, [])
        # Oracle is explicitly post-hoc GT analysis and never a deployable selector.
        gt_candidates = [row for row in candidates if row.get("C_GT_proj") is not None]
        oracle = min(gt_candidates, key=lambda row: float(row["C_GT_proj"])) if gt_candidates else None
        oracle_rows.append({
            "sample_id": state["sample_id"], "crop_id": crop_id,
            "oracle_available": oracle is not None,
            "oracle_ring_id": oracle.get("ring_id") if oracle else None,
            "oracle_deleted_h": oracle.get("deleted_h") if oracle else None,
            "oracle_projective_error_change": oracle.get("C_GT_proj") if oracle else None,
            "oracle_improves": bool(oracle and float(oracle["C_GT_proj"]) < 0.0),
            "may_change_final_coordinate": False,
        })
        for row in candidates:
            protected = any(role in row.get("roles", []) for role in
                            ("DIRECTION_CRITICAL", "RANGE_CRITICAL", "MODE_EXCLUSION"))
            conflict_pass = float(row.get("C50", 0.0)) > 0.0 and float(row.get("C_tail", 0.0)) > 0.0
            # Replay-consistent anomaly evidence is not available in the current
            # four-tree data, so no candidate can satisfy all nine v4 clauses.
            suspect_rows.append({"sample_id": state["sample_id"], "crop_id": crop_id,
                                 "ring_id": row["ring_id"], "conflict_gate_pass": conflict_pass,
                                 "protected_critical": protected,
                                 "search_certificate_pass": row.get("search_certificate", {}).get("passed", False),
                                 "replay_consistent_measurement_anomaly": False,
                                 "harmful_suspect": False,
                                 "reason": "FULL_REPLAY_AND_INDEPENDENT_THRESHOLDS_UNAVAILABLE"})

    write_jsonl(args.output_dir / "final_results.jsonl", final_rows)
    write_jsonl(args.output_dir / "oracle_analysis.jsonl", oracle_rows)
    write_jsonl(args.output_dir / "safe_candidate_audit.jsonl", suspect_rows)
    (args.output_dir / "deployment_gate.json").write_text(
        json.dumps(deployment_gate, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "stage": 7, "schema": "ArcPith-GT-v4-stage7", "records": len(final_rows),
        "safe_prune": "OFF", "safe_prune_applied": 0,
        "final_method_counts": dict(Counter(row["final_method"] for row in final_rows)),
        "final_state_counts": dict(Counter(row["geometry_state"] for row in final_rows)),
        "final_execution_degree_counts": dict(Counter(row["execution_degree"] for row in final_rows)),
        "oracle_available": sum(row["oracle_available"] for row in oracle_rows),
        "oracle_improvement_possible": sum(row["oracle_improves"] for row in oracle_rows),
        "oracle_changed_final_coordinate": 0,
        "deployment_gate_reasons": gate_reasons,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "Stage 7 keeps SAFE_PRUNE OFF because four independent trees cannot satisfy the registered "
        "minimum of max(30, n_power), and validation/replay gates are not closed. Every final coordinate "
        "is the Stage-4 high-resolution F-All estimate. F-Oracle is stored only as a theoretical audit.\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if final_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
