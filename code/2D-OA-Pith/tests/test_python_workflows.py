from __future__ import annotations

import json

import pytest

from oapith.data.manifest import load_manifest
from oapith.workflows import run_calibration, run_consensus


def test_programmatic_consensus_workflow(tmp_path) -> None:
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "consensus.jsonl"
    row = {
        "sample_id": "s1",
        "group_id": "tree1",
        "split": "train",
        "image": "unused.png",
        "curves": [
            {
                "ring_id": 1,
                "arc_id": "a1",
                "annotator_id": "rater_a",
                "points_px": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            },
            {
                "ring_id": 1,
                "arc_id": "a1",
                "annotator_id": "rater_b",
                "points_px": [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            },
        ],
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    run_consensus(source, output)

    records = load_manifest(output)
    assert len(records[0].curves) == 1
    assert records[0].curves[0].annotator_id == "consensus"


def test_programmatic_calibration_workflow(tmp_path) -> None:
    prediction_template = {
        "conformal": {
            "prediction": {
                "components": [
                    {
                        "kind": "near",
                        "weight": 1.0,
                        "mean": [0.0, 0.0],
                        "covariance": [[1.0, 0.0], [0.0, 1.0]],
                        "precision": None,
                    }
                ],
                "null_probability": 0.0,
                "infinity_rho_scale": 0.05,
                "metadata": {},
            }
        }
    }
    prediction_a = tmp_path / "prediction_a.json"
    prediction_b = tmp_path / "prediction_b.json"
    prediction_a.write_text(json.dumps(prediction_template), encoding="utf-8")
    prediction_b.write_text(json.dumps(prediction_template), encoding="utf-8")
    manifest = tmp_path / "calibration.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prediction_json": str(prediction_a),
                        "truth_normalized": [0.0, 0.0],
                        "group_id": "tree_a",
                    }
                ),
                json.dumps(
                    {
                        "prediction_json": str(prediction_b),
                        "truth_normalized": [1.0, 0.0],
                        "group_id": "tree_b",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "conformal.json"

    calibrator = run_calibration(manifest, output, alpha=0.5, group_reduction="max")

    assert calibrator.number_groups == 2
    assert calibrator.threshold == pytest.approx(0.5)
    assert output.exists()
