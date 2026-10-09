#!/usr/bin/env python3
"""Build the frozen parent-section scale table (R_F / w_ref / w_max) for position metrics.

Scales are computed once per section from the full-section GT annotations and
must never be recomputed from predictions (shusui.txt). Outputs a JSONL table
plus a frozen config whose sha256 is recorded in every artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from racpith.evaluation.scales import DEFAULT_PARAMS, compute_section_scales
from racpith.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)


def canonical_sha256(payload: dict[str, Any]) -> str:
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="manifest_grayscale.jsonl (all crops)")
    parser.add_argument("--output", required=True, help="output root (scales/ written under it)")
    args = parser.parse_args()

    output_root = Path(args.output).expanduser().resolve()
    scales_dir = output_root / "scales"
    scales_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.manifest)
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_section[str(row["section_id"])].append(row)

    config = {
        "schema_version": "racpith.section_scales_config.v1",
        "purpose": "frozen parent-section scales for position-metric evaluation only",
        "boundary": "scale table enters the evaluator only; never localization inference",
        "params": dict(DEFAULT_PARAMS),
    }
    config_hash = canonical_sha256(config)
    config_path = scales_dir / "scales_config.json"
    atomic_write_json(config_path, {**config, "config_sha256": config_hash})

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for section_id in sorted(by_section):
        rows = by_section[section_id]
        pith_full = np.asarray([r["pith_full_px"] for r in rows], dtype=np.float64)
        max_disagree = float(np.abs(pith_full - pith_full[0]).max())
        if max_disagree > 1e-6:
            failures.append(
                {
                    "section_id": section_id,
                    "reason": "pith_full_px inconsistent across crops",
                    "max_disagree_px": max_disagree,
                }
            )
            continue
        ann_path = rows[0]["metadata"]["source_full_annotation"]
        scale = compute_section_scales(ann_path, DEFAULT_PARAMS)
        if scale is None:
            failures.append({"section_id": section_id, "reason": "no usable ring polygons"})
            continue
        tree_ids = {derived for r in rows for derived in [str(r["tree_id"])]}
        if len(tree_ids) != 1:
            failures.append(
                {"section_id": section_id, "reason": f"multiple tree ids {sorted(tree_ids)}"}
            )
            continue
        records.append(
            {
                **scale,
                "section_id": section_id,
                "tree_id": sorted(tree_ids)[0],
                "n_crops": len(rows),
                "pith_full_px": [float(v) for v in pith_full[0]],
                "config_sha256": config_hash,
                "source_manifest_sha256": sha256_file(args.manifest),
            }
        )

    if failures:
        raise SystemExit(f"section scale failures: {failures[:10]}")

    scales_path = scales_dir / "section_scales.jsonl"
    atomic_write_jsonl(scales_path, records)
    r_fs = [r["R_F_px"] for r in records]
    w_refs = [r["w_ref_px"] for r in records if r["w_ref_px"] is not None]
    summary = {
        "schema_version": "racpith.section_scales_summary.v1",
        "config_sha256": config_hash,
        "params": dict(DEFAULT_PARAMS),
        "n_sections": len(records),
        "n_crops": sum(r["n_crops"] for r in records),
        "R_F_px": {
            "min": float(np.min(r_fs)),
            "median": float(np.median(r_fs)),
            "max": float(np.max(r_fs)),
        },
        "w_ref_px": (
            {
                "min": float(np.min(w_refs)),
                "median": float(np.median(w_refs)),
                "p90": float(np.percentile(w_refs, 90)),
                "max": float(np.max(w_refs)),
            }
            if w_refs
            else None
        ),
        "w_max_px": {
            "median": float(np.median([r["w_max_px"] for r in records if r["w_max_px"] is not None]))
        },
        "section_scales_sha256": sha256_file(scales_path),
        "scales_config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(args.manifest),
    }
    atomic_write_json(scales_dir / "summary.json", summary)
    print(
        f"wrote {len(records)} section scales -> {scales_path}\n"
        f"R_F median {summary['R_F_px']['median']:.1f} px | "
        f"w_ref median {summary['w_ref_px']['median']:.2f} px | "
        f"config {config_hash[:12]}"
    )


if __name__ == "__main__":
    main()
