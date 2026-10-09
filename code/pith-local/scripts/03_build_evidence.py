#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from racpith.config import load_config
from racpith.contracts import EvidenceBundle
from racpith.evidence import build_evidence
from racpith.provenance import atomic_write_jsonl, is_sha256, read_jsonl, sha256_file


def _validate_crop_inputs(row: Mapping[str, Any]) -> None:
    crop_id = str(row["crop_id"])
    for path_field, hash_field in (
        ("crop_image_path", "crop_image_sha256"),
        ("crop_annotation_path", "crop_annotation_sha256"),
    ):
        path = Path(str(row[path_field])).expanduser()
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"crop input path is invalid: {crop_id}/{path_field}={path}")
        if not is_sha256(row.get(hash_field)):
            raise ValueError(f"crop input hash is invalid: {crop_id}/{hash_field}")
        if sha256_file(path) != row[hash_field]:
            raise ValueError(f"crop input content differs from manifest: {crop_id}/{path_field}")


def _one(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    config_hash: str,
    output_dir: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    crop_id = str(row["crop_id"])
    destination = Path(output_dir) / "per_crop"
    npz_path = destination / f"{crop_id}.npz"
    metadata_path = destination / f"{crop_id}.json"
    try:
        bundle = build_evidence(
            row["crop_annotation_path"],
            row["crop_image_path"],
            config,
            config_hash=config_hash,
            crop_annotation_sha256=str(row["crop_annotation_sha256"]),
            crop_image_sha256=str(row["crop_image_sha256"]),
        )
        # Close the read/check race: both inputs must still name the content that
        # produced the bundle immediately before it is committed and indexed.
        _validate_crop_inputs(row)
        expected_size = tuple(int(value) for value in row["crop_size_px"])
        expected_origin = tuple(float(value) for value in row["crop_origin_source_px"])
        expected_scale = float(row["normalization_scale_px"])
        if (
            bundle.crop_id != crop_id
            or bundle.tree_id != str(row["tree_id"])
            or bundle.section_id != str(row["section_id"])
            or bundle.crop_size_px != expected_size
            or bundle.crop_origin_source_px != expected_origin
            or abs(bundle.normalization_scale_px - expected_scale)
            > 1e-10 * max(1.0, expected_scale)
            or bundle.metadata.get("evidence_config_hash") != config_hash
            or bundle.metadata.get("crop_image_sha256")
            != row["crop_image_sha256"]
            or bundle.metadata.get("crop_annotation_sha256")
            != row["crop_annotation_sha256"]
        ):
            raise ValueError(
                "built evidence identity/coordinate scale disagrees with crop manifest"
            )
        bundle.save(npz_path, metadata_path)
        return {
            "schema_version": "racpith.evidence_index.v1",
            "crop_id": crop_id,
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "PASS",
            "config_hash": config_hash,
            "metadata_path": str(metadata_path.resolve()),
            "npz_path": str(npz_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "npz_sha256": sha256_file(npz_path),
            "crop_image_sha256": row["crop_image_sha256"],
            "crop_annotation_sha256": row["crop_annotation_sha256"],
            "n_nodes": bundle.n_nodes,
            "n_parent_rings": len(bundle.ring_ids),
            "runtime_seconds": time.perf_counter() - started,
            "reason": None,
        }
    except Exception as exc:
        return {
            "schema_version": "racpith.evidence_index.v1",
            "crop_id": crop_id,
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "FAIL",
            "config_hash": config_hash,
            "metadata_path": None,
            "npz_path": None,
            "metadata_sha256": None,
            "npz_sha256": None,
            "crop_image_sha256": row.get("crop_image_sha256"),
            "crop_annotation_sha256": row.get("crop_annotation_sha256"),
            "n_nodes": 0,
            "n_parent_rings": 0,
            "runtime_seconds": time.perf_counter() - started,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _resume_complete_index(
    index_path: Path,
    output_root: Path,
    manifest_rows: list[dict[str, Any]],
    config_hash: str,
) -> list[dict[str, Any]] | None:
    if not index_path.is_file():
        return None
    manifest = {str(row["crop_id"]): row for row in manifest_rows}
    existing: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(index_path):
        if row.get("schema_version") != "racpith.evidence_index.v1":
            raise ValueError("refusing evidence resume: unsupported index schema")
        crop_id = str(row["crop_id"])
        if crop_id in existing:
            raise ValueError(f"refusing evidence resume: duplicate crop {crop_id}")
        if row.get("config_hash") != config_hash:
            raise ValueError(f"refusing evidence resume: config differs for {crop_id}")
        existing[crop_id] = row
    if set(existing) != set(manifest):
        raise ValueError("refusing evidence resume: manifest scope differs from existing index")
    for crop_id, row in existing.items():
        source = manifest[crop_id]
        if any(
            str(row.get(field)) != str(source.get(field))
            for field in (
                "tree_id",
                "section_id",
                "split",
                "crop_image_sha256",
                "crop_annotation_sha256",
            )
        ):
            raise ValueError(f"refusing evidence resume: lineage differs for {crop_id}")
        if not bool(source.get("eligible", True)):
            if row.get("status") != "MANIFEST_INELIGIBLE":
                return None
            continue
        if row.get("status") != "PASS":
            # A recorded FAIL is deterministic under the same frozen
            # configuration and inputs.  Treating it as final keeps resume
            # cheap: retrying failures must not force a rebuild of every
            # intact PASS row (which would also invalidate downstream
            # per-crop resume through rewritten artifact hashes).  Delete
            # evidence_index.jsonl explicitly to retry recorded failures.
            if row.get("status") != "FAIL" or not row.get("reason"):
                return None
            continue
        if (
            sha256_file(source["crop_image_path"])
            != source["crop_image_sha256"]
            or sha256_file(source["crop_annotation_path"])
            != source["crop_annotation_sha256"]
        ):
            raise ValueError(
                f"refusing evidence resume: crop input content changed for {crop_id}"
            )
        metadata = Path(str(row.get("metadata_path"))).expanduser().resolve()
        npz = Path(str(row.get("npz_path"))).expanduser().resolve()
        if not metadata.is_relative_to(output_root) or not npz.is_relative_to(output_root):
            raise ValueError(f"refusing evidence resume: artifact path escapes run root for {crop_id}")
        if (
            sha256_file(metadata) != row.get("metadata_sha256")
            or sha256_file(npz) != row.get("npz_sha256")
        ):
            raise ValueError(f"refusing evidence resume: artifact hash changed for {crop_id}")
        bundle = EvidenceBundle.load(metadata)
        if (
            bundle.crop_id != crop_id
            or bundle.tree_id != str(source["tree_id"])
            or bundle.section_id != str(source["section_id"])
            or bundle.crop_size_px
            != tuple(int(value) for value in source["crop_size_px"])
            or bundle.crop_origin_source_px
            != tuple(float(value) for value in source["crop_origin_source_px"])
            or abs(
                bundle.normalization_scale_px
                - float(source["normalization_scale_px"])
            )
            > 1e-10 * max(1.0, float(source["normalization_scale_px"]))
            or bundle.metadata.get("evidence_config_hash") != config_hash
            or bundle.metadata.get("crop_image_sha256")
            != source["crop_image_sha256"]
            or bundle.metadata.get("crop_annotation_sha256")
            != source["crop_annotation_sha256"]
        ):
            raise ValueError(f"refusing evidence resume: bundle lineage differs for {crop_id}")
    return [existing[crop_id] for crop_id in sorted(existing)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate-independent RAC-Pith evidence")
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on-any-error", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")

    frozen = load_config(args.config)
    rows = read_jsonl(args.crop_manifest)
    seen_crop_ids: set[str] = set()
    for row in rows:
        if row.get("schema_version") != "racpith.crop_manifest.v1":
            raise ValueError("unsupported crop-manifest schema")
        crop_id = str(row["crop_id"])
        if crop_id in seen_crop_ids:
            raise ValueError(f"duplicate crop-manifest row for {crop_id}")
        seen_crop_ids.add(crop_id)
        for field in (
            "tree_id",
            "section_id",
            "split",
            "crop_image_path",
            "crop_annotation_path",
            "crop_origin_source_px",
            "crop_size_px",
            "normalization_scale_px",
            "crop_image_sha256",
            "crop_annotation_sha256",
            "eligible",
        ):
            if field not in row:
                raise ValueError(f"crop {crop_id} is missing manifest field {field}")
        if not isinstance(row["eligible"], bool):
            raise ValueError(f"crop {crop_id} eligible must be a JSON boolean")
        if len(row["crop_size_px"]) != 2 or len(row["crop_origin_source_px"]) != 2:
            raise ValueError(f"crop {crop_id} has invalid coordinate tuple dimensions")
        width, height = (int(value) for value in row["crop_size_px"])
        if width <= 0 or height <= 0:
            raise ValueError(f"crop {crop_id} has non-positive dimensions")
        scale = float(row["normalization_scale_px"])
        if not math.isfinite(scale) or not math.isclose(
            scale,
            math.hypot(width, height),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"crop {crop_id} normalization scale is inconsistent")
        if not all(
            math.isfinite(float(value)) for value in row["crop_origin_source_px"]
        ):
            raise ValueError(f"crop {crop_id} has a non-finite source origin")
        for field in ("crop_image_sha256", "crop_annotation_sha256"):
            if not is_sha256(row[field]):
                raise ValueError(f"crop {crop_id} has invalid {field}")
    if args.split:
        allowed = set(args.split)
        rows = [row for row in rows if row["split"] in allowed]
    if not rows:
        raise ValueError("selected crop-manifest scope is empty")
    for row in rows:
        _validate_crop_inputs(row)
    output = Path(args.output)
    (output / "per_crop").mkdir(parents=True, exist_ok=True)
    index_path = output / "evidence_index.jsonl"
    if args.resume:
        resumed = _resume_complete_index(
            index_path,
            output.expanduser().resolve(),
            rows,
            frozen.sha256,
        )
        if resumed is not None:
            ineligible_count = sum(
                row["status"] == "MANIFEST_INELIGIBLE" for row in resumed
            )
            failed_count = sum(row["status"] == "FAIL" for row in resumed)
            passed = sum(row["status"] == "PASS" for row in resumed)
            print(
                f"evidence: resumed {passed} PASS, {failed_count} recorded FAIL, "
                f"{ineligible_count} manifest-ineligible; "
                f"config={frozen.sha256}"
            )
            return
    ineligible = [row for row in rows if not bool(row.get("eligible", True))]
    rows = [row for row in rows if bool(row.get("eligible", True))]
    results: list[dict[str, Any]] = [
        {
            "schema_version": "racpith.evidence_index.v1",
            "crop_id": str(row["crop_id"]),
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "MANIFEST_INELIGIBLE",
            "config_hash": frozen.sha256,
            "metadata_path": None,
            "npz_path": None,
            "metadata_sha256": None,
            "npz_sha256": None,
            "crop_image_sha256": row["crop_image_sha256"],
            "crop_annotation_sha256": row["crop_annotation_sha256"],
            "n_nodes": 0,
            "n_parent_rings": 0,
            "runtime_seconds": 0.0,
            "reason": "MANIFEST_INELIGIBLE: "
            + "|".join(str(value) for value in row.get("rejection_reasons", [])),
        }
        for row in ineligible
    ]
    if args.workers <= 1:
        results.extend(_one(row, frozen.data, frozen.sha256, str(output)) for row in rows)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(_one, row, frozen.data, frozen.sha256, str(output))
                for row in rows
            ]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda row: row["crop_id"])
    atomic_write_jsonl(index_path, results)
    failures = sum(row["status"] == "FAIL" for row in results)
    ineligible_count = sum(
        row["status"] == "MANIFEST_INELIGIBLE" for row in results
    )
    passed = sum(row["status"] == "PASS" for row in results)
    print(
        f"evidence: {passed} PASS, {failures} FAIL, "
        f"{ineligible_count} manifest-ineligible; config={frozen.sha256}"
    )
    if failures and args.fail_on_any_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
