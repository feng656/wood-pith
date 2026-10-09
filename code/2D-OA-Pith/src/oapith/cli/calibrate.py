from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from oapith.calibration import ConformalCalibrator, MixturePrediction
from oapith.calibration.conformal import ConformalComponent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a disc/tree-grouped conformal threshold")
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSONL with prediction_json, truth_normalized, group_id",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--group-reduction", choices=["max", "first"], default="max")
    return parser


def _prediction(value: dict[str, object]) -> MixturePrediction:
    return MixturePrediction(
        components=[ConformalComponent(**item) for item in value["components"]],
        null_probability=float(value.get("null_probability", 0.0)),
        infinity_rho_scale=float(value.get("infinity_rho_scale", 0.05)),
        metadata=dict(value.get("metadata", {})),
    )


def run_calibration(
    manifest: str | Path,
    output: str | Path,
    *,
    alpha: float = 0.05,
    group_reduction: str = "max",
) -> ConformalCalibrator:
    """Fit and save a grouped conformal calibrator from Python."""
    predictions, truth, groups = [], [], []
    with Path(manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction_file = json.loads(Path(row["prediction_json"]).read_text(encoding="utf-8"))
            predictions.append(_prediction(prediction_file["conformal"]["prediction"]))
            truth.append(torch.tensor(row["truth_normalized"], dtype=torch.float64))
            groups.append(str(row["group_id"]))
    calibrator = ConformalCalibrator(alpha, reduction=group_reduction)
    calibrator.fit(predictions, truth, groups).save(output)
    return calibrator


def main() -> None:
    args = build_parser().parse_args()
    run_calibration(
        args.manifest,
        args.output,
        alpha=args.alpha,
        group_reduction=args.group_reduction,
    )


if __name__ == "__main__":
    main()
