#!/usr/bin/env python3
"""Rasterize manifest ring curves and build a grayscale-image manifest.

The source crop images are retained as ``image_color``.  ``image`` in the
derived manifest points to a grayscale annotation raster matching the
``dataset_grid`` convention: white background (255), ring ``k`` drawn with
value ``k + 1``, and a 3-pixel polyline.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ANOMALY_GAP_MIN_PX = 50.0
ANOMALY_GAP_RATIO = 4.0
MIN_SPLIT_FRAGMENT_POINTS = 4


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest row {line_no} is not an object")
            rows.append(row)
    return rows


def _ring_value(curve: dict[str, Any]) -> int:
    """Return the stable grayscale value for a curve's parent ring."""
    raw = curve.get("order", curve.get("ring_id"))
    try:
        order = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"curve has no numeric ring order: {raw!r}") from exc
    # 0 is reserved for black only conceptually; 255 is the background.
    # UruDendro4 has far fewer than 253 parent rings, but fail clearly if a
    # future input would collide with the background value.
    value = order + 1
    if not 1 <= value <= 254:
        raise ValueError(f"ring order {order} cannot be encoded in uint8")
    return value


def _split_curve_points(
    points: np.ndarray,
    *,
    min_gap_px: float = ANOMALY_GAP_MIN_PX,
    gap_ratio: float = ANOMALY_GAP_RATIO,
) -> tuple[list[np.ndarray], int]:
    """Split obvious annotation jumps without changing any valid vertices.

    A fixed gap threshold alone would split legitimate sparse curves.  Requiring
    the jump to also be ten times the median spacing of the same curve makes the
    rule target the long cross-connection pattern seen in the source labels.
    """
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return [], 0
    pending = [points]
    pieces: list[np.ndarray] = []
    split_count = 0
    while pending:
        current = pending.pop()
        gaps = np.linalg.norm(np.diff(current, axis=0), axis=1)
        positive = gaps[gaps > 1e-9]
        if not len(positive):
            pieces.append(current)
            continue
        median_gap = float(np.median(positive))
        cuts = np.flatnonzero(
            (gaps >= float(min_gap_px))
            & (gaps >= float(gap_ratio) * max(median_gap, 1e-9))
        ) + 1
        if not len(cuts):
            pieces.append(current)
            continue
        split_count += int(len(cuts))
        pending.extend(
            piece
            for piece in np.split(current, cuts)
            if len(piece) >= MIN_SPLIT_FRAGMENT_POINTS
        )
    return pieces, split_count


def _split_row_curves(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    curves: list[dict[str, Any]] = []
    split_count = 0
    for curve in row.get("curves", []):
        points = np.asarray(curve.get("points_px", curve.get("points", [])), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            continue
        pieces, count = _split_curve_points(points)
        split_count += count
        for fragment_index, piece in enumerate(pieces):
            item = copy.deepcopy(curve)
            item["points_px"] = piece.tolist()
            item["source_fragment_index"] = fragment_index
            item["source_discontinuity_split"] = bool(count)
            item["closed"] = False
            curves.append(item)
    return curves, split_count


def _rasterize(row: dict[str, Any], thickness: int) -> tuple[np.ndarray, int]:
    size = row.get("image_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("image_size must be [width, height]")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image_size: {size!r}")
    canvas = np.full((height, width), 255, dtype=np.uint8)
    curves = row.get("curves")
    if not isinstance(curves, list):
        raise ValueError("manifest row has no curves list")
    drawn = 0
    for curve in curves:
        points = np.asarray(curve.get("points_px", curve.get("points", [])), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            continue
        if not np.all(np.isfinite(points)):
            raise ValueError("curve contains non-finite coordinates")
        # OpenCV uses (x, y), matching the manifest convention.  Rounding is
        # equivalent to the original dataset_grid generator's integer cast
        # while clipping prevents an out-of-bounds annotation from wrapping.
        pts = np.rint(points).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.polylines(canvas, [pts], isClosed=False,
                      color=_ring_value(curve), thickness=thickness,
                      lineType=cv2.LINE_8)
        drawn += 1
    return canvas, drawn


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="directory containing grayscale_dataset and manifest_grayscale.jsonl")
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N rows (0 means all rows)")
    args = parser.parse_args()
    if args.thickness < 1:
        parser.error("--thickness must be positive")

    rows = _read_jsonl(args.manifest)
    if args.limit > 0:
        rows = rows[:args.limit]
    output_root = args.output_root.resolve()
    image_root = output_root / "grayscale_dataset"
    image_root.mkdir(parents=True, exist_ok=True)
    output_manifest = output_root / "manifest_grayscale.jsonl"
    output_summary = output_root / "manifest_grayscale_summary.json"

    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_curves = 0
    total_drawn = 0
    for index, row in enumerate(rows, 1):
        sample_id = str(row.get("sample_id", f"row_{index:06d}"))
        try:
            derived = copy.deepcopy(row)
            derived["curves"], split_count = _split_row_curves(derived)
            raster, drawn = _rasterize(derived, args.thickness)
            image_path = image_root / f"{sample_id}.png"
            if image_path.exists() and image_path.is_dir():
                raise ValueError(f"output path is a directory: {image_path}")
            if not cv2.imwrite(str(image_path), raster):
                raise OSError(f"cv2.imwrite failed: {image_path}")
            derived["image_color"] = str(row["image"])
            derived["image"] = str(image_path)
            metadata = derived.setdefault("metadata", {})
            metadata["grayscale_annotation"] = {
                "convention": "dataset_grid",
                "background_value": 255,
                "ring_value": "order + 1",
                "line_thickness_px": args.thickness,
                "source_color_image": str(row["image"]),
                "source_ring_annotation": metadata.get("source_annotation"),
                "raster_curve_count": drawn,
                "source_discontinuities_split": split_count,
                "anomaly_split_rule": {
                    "minimum_gap_px": ANOMALY_GAP_MIN_PX,
                    "minimum_gap_to_median_ratio": ANOMALY_GAP_RATIO,
                    "minimum_fragment_points": MIN_SPLIT_FRAGMENT_POINTS,
                },
            }
            metadata["source_discontinuities_split"] = split_count
            output_rows.append(derived)
            total_curves += len(derived.get("curves", []))
            total_drawn += drawn
        except Exception as exc:  # keep a complete failure report
            failures.append({"row": index, "sample_id": sample_id,
                             "error": f"{type(exc).__name__}: {exc}"})

    with output_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "source_manifest": str(args.manifest.resolve()),
        "output_manifest": str(output_manifest),
        "output_image_root": str(image_root),
        "requested_rows": len(rows),
        "written_rows": len(output_rows),
        "failed_rows": len(failures),
        "total_source_curves": total_curves,
        "total_drawn_curves": total_drawn,
        "line_thickness_px": args.thickness,
        "pixel_convention": {"background": 255, "ring_value": "order + 1"},
        "anomaly_split_rule": {
            "minimum_gap_px": ANOMALY_GAP_MIN_PX,
            "minimum_gap_to_median_ratio": ANOMALY_GAP_RATIO,
            "minimum_fragment_points": MIN_SPLIT_FRAGMENT_POINTS,
        },
        "failures": failures,
        "manifest_sha256": _hash_file(output_manifest),
    }
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
