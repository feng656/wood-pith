#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from racpith.baselines import BASELINE_METHODS, run_geometric_baselines
from racpith.config import load_config
from racpith.contracts import EvidenceBundle
from racpith.estimator import RacPithEstimator
from racpith.evaluation.metrics import (
    load_evidence_index,
    validate_evidence_bundle_lineage,
)
from racpith.provenance import atomic_write_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B0-B5 geometry-fair baselines")
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    args = parser.parse_args()
    frozen = load_config(args.config)
    estimator = RacPithEstimator(frozen, run_id="baselines")
    rows = list(
        load_evidence_index(
            args.evidence_index,
            expected_config_hash=frozen.sha256,
        ).values()
    )
    if args.split:
        rows = [row for row in rows if row["split"] in set(args.split)]
    if not rows:
        raise ValueError("selected evidence-index scope is empty")
    seen_crops: set[str] = set()
    for row in rows:
        if row.get("schema_version") != "racpith.evidence_index.v1":
            raise ValueError("unsupported evidence-index schema")
        crop_id = str(row["crop_id"])
        if crop_id in seen_crops:
            raise ValueError(f"duplicate evidence-index crop {crop_id}")
        seen_crops.add(crop_id)
    output_rows = []
    for row in rows:
        if row["status"] != "PASS":
            for method in BASELINE_METHODS:
                output_rows.append(
                    {
                        "schema_version": "racpith.baseline.v1",
                        "crop_id": row["crop_id"],
                        "tree_id": row["tree_id"],
                        "section_id": row["section_id"],
                        "split": row["split"],
                        "method": method,
                        "status": "UPSTREAM_EVIDENCE_FAIL",
                        "state": "REJECT",
                        "raw_center_norm": None,
                        "converged": False,
                        "search_adequate": False,
                        "config_hash": frozen.sha256,
                        "reason": row.get("reason"),
                    }
                )
            continue
        try:
            if (
                sha256_file(row["metadata_path"]) != row.get("metadata_sha256")
                or sha256_file(row["npz_path"]) != row.get("npz_sha256")
            ):
                raise ValueError("indexed evidence content has changed")
            bundle = EvidenceBundle.load(row["metadata_path"])
            validate_evidence_bundle_lineage(
                row,
                bundle,
                expected_config_hash=frozen.sha256,
            )
            for record in run_geometric_baselines(bundle, estimator):
                output_rows.append(
                    {
                        **record,
                        "crop_id": bundle.crop_id,
                        "tree_id": bundle.tree_id,
                        "section_id": bundle.section_id,
                        "split": row["split"],
                        "status": str(record.get("status", "FINISHED")),
                        "evidence_metadata_sha256": row.get("metadata_sha256"),
                        "evidence_npz_sha256": row.get("npz_sha256"),
                    }
                )
        except Exception as exc:
            for method in BASELINE_METHODS:
                output_rows.append(
                    {
                        "schema_version": "racpith.baseline.v1",
                        "crop_id": row["crop_id"],
                        "tree_id": row["tree_id"],
                        "section_id": row["section_id"],
                        "split": row["split"],
                        "method": method,
                        "status": "CRASH",
                        "state": "REJECT",
                        "raw_center_norm": None,
                        "converged": False,
                        "search_adequate": False,
                        "config_hash": frozen.sha256,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output, output_rows)
    failures = sum(row.get("status") != "FINISHED" for row in output_rows)
    print(f"baseline records={len(output_rows)}, failures={failures}")


if __name__ == "__main__":
    main()
