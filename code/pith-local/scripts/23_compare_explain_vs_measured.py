#!/usr/bin/env python3
"""Compare geometric sub-arc explanations against measured delete-refit contributions.

For each crop of the validation8 mini-run, joins the subarc contribution records
(level='subarc' from scripts/05_run_contributions.py) to the explanation records
(scripts/22_explain_subarcs.py) by geometric sub-arc midpoint (naming conventions
differ between the two pipelines), then reports per-category statistics of the
measured ``contrib_gt_px`` (error_after_delete - error_all; positive = the
sub-arc helps, negative = it hurts).

Honest-reporting rules:
- subarc-scale delete-refit deltas are known to be tiny (smoke median ~0.01 px);
  the comparison therefore reports sign agreement AND magnitude, and must not
  claim that geometry equals measured contribution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from racpith.provenance import atomic_write_json, atomic_write_jsonl, sha256_file


def load_bundle(crop_id: str, evidence_root: Path) -> dict[str, Any] | None:
    metadata_path = evidence_root / "per_crop" / f"{crop_id}.json"
    npz_path = evidence_root / "per_crop" / f"{crop_id}.npz"
    if not metadata_path.is_file() or not npz_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as arrays:
        return {
            "arc_index": {str(value): index for index, value in enumerate(metadata["arc_ids"])},
            "arc_indices": np.asarray(arrays["arc_index"], dtype=np.int64),
            "arc_fractions": np.asarray(arrays["arc_fraction"], dtype=float),
            "points_crop_px": np.asarray(arrays["points_crop_px"], dtype=float),
        }


def contribution_midpoint(record: dict[str, Any], bundle: dict[str, Any]) -> np.ndarray | None:
    arc_id = str(record.get("arc_id") or "")
    idx = bundle["arc_index"].get(arc_id)
    if idx is None:
        stripped = arc_id.split(":", 1)[1] if ":" in arc_id else None
        idx = bundle["arc_index"].get(stripped) if stripped is not None else None
    if idx is None:
        return None
    mask = bundle["arc_indices"] == int(idx)
    interval = record.get("interval_fraction")
    if interval is not None:
        lo = float(np.clip(interval[0], 0.0, 1.0))
        hi = float(np.clip(interval[1], 0.0, 1.0))
        mask &= bundle["arc_fractions"] >= lo - 1e-12
        mask &= bundle["arc_fractions"] <= hi + 1e-12
    pts = bundle["points_crop_px"][mask]
    return pts.mean(axis=0) if len(pts) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributions", required=True, help="validation8 contributions per_crop dir")
    parser.add_argument("--evidence-root", required=True, help="validation8 evidence root")
    parser.add_argument("--explain-records", required=True, help="subarc_explanations.jsonl from script 22")
    parser.add_argument("--output", required=True, help="output root (validation8 comparison files)")
    args = parser.parse_args()

    contrib_dir = Path(args.contributions).expanduser().resolve()
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # load explanation records for the validation crops only (single pass)
    crop_ids = {path.stem for path in sorted(contrib_dir.glob("*.jsonl"))}
    explain_by_crop: dict[str, list[dict[str, Any]]] = {cid: [] for cid in crop_ids}
    for line in Path(args.explain_records).open(encoding="utf-8"):
        rec = json.loads(line)
        cid = str(rec["crop_id"])
        if cid in explain_by_crop:
            explain_by_crop[cid].append(rec)
    for crop_id in sorted(crop_ids):
        path = contrib_dir / f"{crop_id}.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        subarc_rows = [r for r in rows if r.get("level") == "subarc"]
        if not subarc_rows:
            continue
        bundle = load_bundle(crop_id, evidence_root)
        if bundle is None:
            print(f"WARN: no evidence bundle for {crop_id}")
            continue
        exp_mid: list[np.ndarray] = [
            np.asarray(rec["points_px"], dtype=float).mean(axis=0) for rec in explain_by_crop[crop_id]
        ]
        pairs: list[dict[str, Any]] = []
        matched = 0
        for row in subarc_rows:
            mid = contribution_midpoint(row, bundle)
            if mid is None or not exp_mid:
                continue
            dists = np.linalg.norm(np.asarray(exp_mid) - mid[None, :], axis=1)
            j = int(dists.argmin())
            if dists[j] > 8.0:
                continue
            matched += 1
            exp = explain_by_crop[crop_id][j]
            contrib = row.get("contrib_gt_px")
            shift = row.get("shift_px")
            sign_ok = None
            if contrib is not None:
                if exp["category"] in {"DEVIATING"}:
                    sign_ok = contrib < 0.0
                elif exp["category"] in {"SUPPORTING", "SUPPORTING_SHORT"}:
                    sign_ok = contrib > 0.0
                elif exp["category"] in {"BOUNDARY_TRUNCATED"}:
                    sign_ok = abs(contrib) <= 1.0
            pairs.append(
                {
                    "crop_id": crop_id,
                    "ring_id": str(exp["ring_id"]),
                    "fragment_id": str(exp["fragment_id"]),
                    "interval_fraction": row.get("interval_fraction"),
                    "category": exp["category"],
                    "agreement": exp["agreement"],
                    "contrib_gt_px": contrib,
                    "shift_px": shift,
                    "gt_label": row.get("gt_label"),
                    "sign_consistent_with_hypothesis": sign_ok,
                }
            )
        atomic_write_jsonl(out / f"{crop_id}.jsonl", pairs)
        print(f"{crop_id}: matched {matched}/{len(subarc_rows)} subarcs")

    # dataset-level summary over the validation crops
    all_pairs = []
    for path in sorted(out.glob("*.jsonl")):
        all_pairs.extend(json.loads(line) for line in path.open(encoding="utf-8"))
    by_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted({p["category"] for p in all_pairs}):
        sel = [p for p in all_pairs if p["category"] == cat]
        vals = np.asarray([p["contrib_gt_px"] for p in sel if p["contrib_gt_px"] is not None])
        signed = [p for p in sel if p["sign_consistent_with_hypothesis"] is not None]
        by_cat[cat] = {
            "n": len(sel),
            "contrib_gt_px": {
                "median": float(np.median(vals)) if len(vals) else None,
                "p90_abs": float(np.percentile(np.abs(vals), 90)) if len(vals) else None,
                "frac_abs_lt_1px": float(np.mean(np.abs(vals) < 1.0)) if len(vals) else None,
            },
            "sign_consistency_rate": float(
                np.mean([p["sign_consistent_with_hypothesis"] for p in signed])
            )
            if signed
            else None,
        }
    summary = {
        "schema_version": "racpith.explain_vs_measured.v1",
        "n_crops": len({p["crop_id"] for p in all_pairs}),
        "n_matched_subarcs": len(all_pairs),
        "by_category": by_cat,
        "boundaries": [
            "subarc-scale delete-refit deltas are tiny; sign consistency is directional only",
            "geometry is diagnostics; it does not replace measured contributions",
        ],
        "input_hashes": {
            "explain_records": sha256_file(args.explain_records),
        },
    }
    atomic_write_json(out / "comparison_summary.json", summary)
    print(json.dumps(summary["by_category"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
