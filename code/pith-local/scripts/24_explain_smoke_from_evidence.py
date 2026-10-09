#!/usr/bin/env python3
"""Generate sub-arc explanation records for a legacy smoke crop from its evidence bundle.

The smoke crop (T0_B1_N32_ADAP_s800_x400_y400) is not part of the 5182-crop
dataset, so scripts/22_explain_subarcs.py cannot cover it. This script builds
equivalent explanation records directly from the smoke evidence npz (exact
sampled quadrature nodes) so scripts/23_compare_explain_vs_measured.py can
compare geometric explanations against the 190 measured sub-arc delete-refits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from racpith.evaluation.subarc_explain import explain_from_evidence
from racpith.provenance import atomic_write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, help="smoke evidence root")
    parser.add_argument("--predictions", required=True, help="smoke core/predictions dir")
    parser.add_argument("--crop-id", required=True)
    parser.add_argument("--image-size", required=True, help="W,H in px")
    parser.add_argument("--max-ring", type=int, required=True, help="largest ring label + 1 for zone logic")
    parser.add_argument("--output", required=True, help="output jsonl")
    args = parser.parse_args()

    evidence_root = Path(args.evidence_root).expanduser().resolve()
    metadata_path = evidence_root / "per_crop" / f"{args.crop_id}.json"
    npz_path = evidence_root / "per_crop" / f"{args.crop_id}.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as arrays:
        bundle = {
            "arc_index": {str(value): index for index, value in enumerate(metadata["arc_ids"])},
            "arc_indices": np.asarray(arrays["arc_index"], dtype=np.int64),
            "arc_fractions": np.asarray(arrays["arc_fraction"], dtype=float),
            "points_crop_px": np.asarray(arrays["points_crop_px"], dtype=float),
            "tangents": np.asarray(arrays["tangents"], dtype=float),
        }
    pred_path = Path(args.predictions).expanduser().resolve() / f"{args.crop_id}.json"
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    center = pred.get("raw_center_crop_px")
    width, height = (float(v) for v in args.image_size.split(","))
    records = explain_from_evidence(
        bundle,
        crop_id=args.crop_id,
        image_size=[width, height],
        final_center=np.asarray(center, dtype=float) if center is not None else None,
        max_ring=int(args.max_ring),
    )
    atomic_write_jsonl(args.output, records)
    print(f"wrote {len(records)} smoke explain records -> {args.output}")


if __name__ == "__main__":
    main()
