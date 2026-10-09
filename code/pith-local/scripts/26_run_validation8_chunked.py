#!/usr/bin/env python3
"""Run the validation8 subarc contributions one crop at a time with a timeout.

Rationale: scripts/05_run_contributions.py processes crops sequentially; one
pathological crop (refit hang) stalls the whole batch. This driver runs 05 once
per crop with inputs pruned to that single crop (evidence index + evidence
files + prediction + result-index row), a per-crop wall-clock budget, and
skips crops whose per-crop record already exists.

Lineage is preserved: the crop manifest stays the full 8-crop manifest (so the
evidence-orphan check passes), while evidence/predictions/result-index are
pruned to the current crop, which makes `joined` contain exactly one crop.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from racpith.provenance import atomic_write_json, atomic_write_jsonl, sha256_file

PYTHON = r"D:\Anaconda3\Anaconda3\envs\py311\python.exe"
SCRIPT_05 = Path(__file__).resolve().parent / "05_run_contributions.py"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def build_chunk_inputs(
    crop_id: str,
    chunk_root: Path,
    *,
    manifest_path: Path,
    evidence_root: Path,
    evidence_index_path: Path,
    predictions_root: Path,
    result_index_path: Path,
) -> dict[str, Path]:
    """Copy/prune the single-crop inputs under chunk_root; return their paths."""
    chunk_root.mkdir(parents=True, exist_ok=True)

    # evidence: index row + per_crop files for this crop only
    ev_root = chunk_root / "evidence"
    ev_per_crop = ev_root / "per_crop"
    ev_per_crop.mkdir(parents=True, exist_ok=True)
    index_rows = [r for r in read_jsonl(evidence_index_path) if str(r["crop_id"]) == crop_id]
    if len(index_rows) != 1:
        raise ValueError(f"evidence index has {len(index_rows)} rows for {crop_id}")
    # rewrite artifact paths to the chunk dir (content hashes stay identical)
    row = dict(index_rows[0])
    for field in ("metadata_path", "npz_path"):
        row[field] = str((ev_per_crop / f"{crop_id}{Path(str(row[field])).suffix}").resolve())
    ev_index = ev_root / "evidence_index.jsonl"
    with ev_index.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for suffix in (".json", ".npz"):
        src = evidence_root / "per_crop" / f"{crop_id}{suffix}"
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, ev_per_crop / f"{crop_id}{suffix}")

    # predictions: single prediction file + pruned result index with rewritten path
    pred_root = chunk_root / "predictions"
    src_pred = _find_prediction(predictions_root, crop_id)
    dst_pred = pred_root / f"{crop_id}.json"
    dst_pred.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pred, dst_pred)
    ri_rows = [r for r in read_jsonl(result_index_path) if str(r["crop_id"]) == crop_id]
    if len(ri_rows) != 1:
        raise ValueError(f"result index has {len(ri_rows)} rows for {crop_id}")
    ri_row = dict(ri_rows[0])
    ri_row["prediction_path"] = str(dst_pred.resolve())
    ri_path = chunk_root / "result_index.jsonl"
    atomic_write_jsonl(ri_path, [ri_row])
    return {
        "evidence_index": ev_index,
        "predictions": pred_root,
        "result_index": ri_path,
        "manifest": manifest_path,
    }


def _find_prediction(predictions_root: Path, crop_id: str) -> Path:
    for path in sorted(predictions_root.rglob(f"{crop_id}.json")):
        return path
    raise FileNotFoundError(f"prediction for {crop_id} under {predictions_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-manifest", required=True, help="full 8-crop manifest")
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="shared contributions root")
    parser.add_argument("--chunk-root", required=True, help="temporary per-crop chunk dir")
    parser.add_argument("--timeout", type=int, default=1500, help="per-crop wall-clock budget (s)")
    parser.add_argument("--log", required=True, help="progress log file")
    args = parser.parse_args()

    manifest_path = Path(args.crop_manifest).expanduser().resolve()
    evidence_index_path = Path(args.evidence_index).expanduser().resolve()
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    predictions_root = Path(args.predictions).expanduser().resolve()
    result_index_path = Path(args.result_index).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    chunk_root = Path(args.chunk_root).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")

    manifest = {str(r["crop_id"]): r for r in read_jsonl(manifest_path)}
    done_root = output / "per_crop"
    results: list[dict[str, Any]] = []
    for crop_id in sorted(manifest):
        record_path = done_root / f"{crop_id}.jsonl"
        if record_path.is_file():
            results.append({"crop_id": crop_id, "status": "SKIPPED_EXISTS"})
            log(f"{crop_id}: skipped (per_crop record exists)")
            continue
        chunk = chunk_root / crop_id
        if chunk.exists():
            shutil.rmtree(chunk)
        inputs = build_chunk_inputs(
            crop_id,
            chunk,
            manifest_path=manifest_path,
            evidence_root=evidence_root,
            evidence_index_path=evidence_index_path,
            predictions_root=predictions_root,
            result_index_path=result_index_path,
        )
        chunk_out = chunk / "contributions"
        command = [
            PYTHON,
            str(SCRIPT_05),
            "--evidence-index",
            str(inputs["evidence_index"]),
            "--crop-manifest",
            str(inputs["manifest"]),
            "--predictions",
            str(inputs["predictions"]),
            "--result-index",
            str(inputs["result_index"]),
            "--config",
            str(config_path),
            "--output",
            str(chunk_out),
        ]
        started = time.time()
        log(f"{crop_id}: starting")
        try:
            proc = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            elapsed = time.time() - started
            tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            if proc.returncode != 0:
                results.append({"crop_id": crop_id, "status": "FAILED", "runtime_s": round(elapsed)})
                log(f"{crop_id}: FAILED rc={proc.returncode} after {elapsed:.0f}s: {tail[:200]}")
                continue
            src_record = chunk_out / "per_crop" / f"{crop_id}.jsonl"
            if not src_record.is_file():
                results.append({"crop_id": crop_id, "status": "NO_RECORD", "runtime_s": round(elapsed)})
                log(f"{crop_id}: finished but no per_crop record written")
                continue
            done_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_record, record_path)
            results.append({"crop_id": crop_id, "status": "PASS", "runtime_s": round(elapsed)})
            log(f"{crop_id}: PASS after {elapsed:.0f}s")
        except subprocess.TimeoutExpired:
            elapsed = time.time() - started
            results.append({"crop_id": crop_id, "status": "TIMEOUT", "runtime_s": round(elapsed)})
            log(f"{crop_id}: TIMEOUT after {args.timeout}s")
    summary = {
        "schema_version": "racpith.validation8_chunked.v1",
        "n_crops": len(results),
        "statuses": {
            status: sum(1 for r in results if r["status"] == status)
            for status in {r["status"] for r in results}
        },
        "crops": results,
        "inputs": {
            "crop_manifest": sha256_file(manifest_path),
            "evidence_index": sha256_file(evidence_index_path),
            "result_index": sha256_file(result_index_path),
            "config": sha256_file(config_path),
        },
    }
    atomic_write_json(output / "chunked_run_summary.json", summary)
    log(f"ALL DONE: {summary['statuses']}")


if __name__ == "__main__":
    main()
