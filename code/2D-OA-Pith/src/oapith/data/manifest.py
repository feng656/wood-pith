from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class RingCurve:
    ring_id: int
    points_px: list[list[float]]
    sigma_px: float | list[float] = 1.0
    visibility: float = 1.0
    annotator_id: str | None = None
    arc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.points_px) < 3 or any(len(point) != 2 for point in self.points_px):
            raise ValueError("every ring curve needs at least three [x,y] points")
        if any(not math.isfinite(float(value)) for point in self.points_px for value in point):
            raise ValueError("ring points must all be finite")
        if any(
            first[0] == second[0] and first[1] == second[1]
            for first, second in zip(self.points_px, self.points_px[1:])
        ):
            raise ValueError("ring curves may not contain consecutive duplicate points")
        if isinstance(self.sigma_px, list) and len(self.sigma_px) != len(self.points_px):
            raise ValueError("per-vertex sigma_px must match points_px")
        sigma_values = self.sigma_px if isinstance(self.sigma_px, list) else [self.sigma_px]
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in sigma_values):
            raise ValueError("sigma_px must be finite and strictly positive")
        if not math.isfinite(self.visibility):
            raise ValueError("visibility must be finite")
        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError("visibility must be in [0,1]")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RingCurve":
        known = {
            "ring_id",
            "points_px",
            "sigma_px",
            "visibility",
            "annotator_id",
            "arc_id",
            "metadata",
        }
        curve = cls(
            ring_id=int(value["ring_id"]),
            points_px=value["points_px"],
            sigma_px=value.get("sigma_px", 1.0),
            visibility=float(value.get("visibility", 1.0)),
            annotator_id=value.get("annotator_id"),
            arc_id=value.get("arc_id"),
            metadata={
                **dict(value.get("metadata", {})),
                **{key: item for key, item in value.items() if key not in known},
            },
        )
        curve.validate()
        return curve


@dataclass
class SampleRecord:
    sample_id: str
    image: str
    group_id: str
    curves: list[RingCurve]
    pith_px: list[float] | None = None
    pith_cov_px: list[list[float]] | None = None
    pith_state: int | None = None
    rings_annotated: bool = True
    mm_per_pixel: list[float] | None = None
    defects: list[dict[str, Any]] = field(default_factory=list)
    unknown_mask: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.pith_px is not None and len(self.pith_px) != 2:
            raise ValueError("pith_px must be [x,y] and may lie outside the image")
        if self.pith_px is not None and any(
            not math.isfinite(float(value)) for value in self.pith_px
        ):
            raise ValueError("pith_px must be finite")
        if self.pith_cov_px is not None:
            if len(self.pith_cov_px) != 2 or any(len(row) != 2 for row in self.pith_cov_px):
                raise ValueError("pith_cov_px must be 2x2")
            covariance = [[float(value) for value in row] for row in self.pith_cov_px]
            if any(not math.isfinite(value) for row in covariance for value in row):
                raise ValueError("pith_cov_px must be finite")
            if abs(covariance[0][1] - covariance[1][0]) > 1e-8:
                raise ValueError("pith_cov_px must be symmetric")
            determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] ** 2
            if covariance[0][0] < 0 or covariance[1][1] < 0 or determinant < -1e-10:
                raise ValueError("pith_cov_px must be positive semidefinite")
        if self.pith_state is not None and (
            not isinstance(self.pith_state, int) or self.pith_state not in {0, 1, 2, 3}
        ):
            raise ValueError("pith_state must be one of 0,1,2,3")
        if self.mm_per_pixel is not None:
            if (
                len(self.mm_per_pixel) != 2
                or any(not math.isfinite(float(value)) for value in self.mm_per_pixel)
                or min(self.mm_per_pixel) <= 0
            ):
                raise ValueError("mm_per_pixel must contain two positive spacings")
        arc_keys = [
            (curve.arc_id, curve.annotator_id)
            for curve in self.curves
            if curve.arc_id is not None
        ]
        if len(arc_keys) != len(set(arc_keys)):
            raise ValueError("(arc_id, annotator_id) pairs must be unique within a sample")
        for curve in self.curves:
            curve.validate()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SampleRecord":
        known = {
            "sample_id",
            "image",
            "group_id",
            "curves",
            "pith_px",
            "pith_cov_px",
            "pith_state",
            "rings_annotated",
            "mm_per_pixel",
            "defects",
            "unknown_mask",
            "split",
            "metadata",
        }
        record = cls(
            sample_id=str(value["sample_id"]),
            image=str(value["image"]),
            group_id=str(value["group_id"]),
            curves=[RingCurve.from_dict(item) for item in value.get("curves", [])],
            pith_px=value.get("pith_px"),
            pith_cov_px=value.get("pith_cov_px"),
            pith_state=value.get("pith_state"),
            rings_annotated=bool(value.get("rings_annotated", True)),
            mm_per_pixel=value.get("mm_per_pixel"),
            defects=value.get("defects", []),
            unknown_mask=value.get("unknown_mask"),
            split=value.get("split"),
            metadata={
                **dict(value.get("metadata", {})),
                **{key: item for key, item in value.items() if key not in known},
            },
        )
        record.validate()
        return record


def load_manifest(path: str | Path, split: str | None = None) -> list[SampleRecord]:
    all_records: list[SampleRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = SampleRecord.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
            all_records.append(record)
    sample_ids = [record.sample_id for record in all_records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be globally unique")
    group_splits: dict[str, set[str | None]] = {}
    for record in all_records:
        group_splits.setdefault(record.group_id, set()).add(record.split)
    leaking = {group: values for group, values in group_splits.items() if len(values) > 1}
    if leaking:
        raise ValueError(f"group_id values cross dataset splits: {leaking}")
    records = [
        record for record in all_records if split is None or record.split == split
    ]
    if not records:
        raise ValueError(f"no records found in {path!s} for split={split!r}")
    return records


def dump_manifest(records: Iterable[SampleRecord], path: str | Path) -> None:
    """Write JSONL while preserving out-of-frame coordinates without clipping."""
    from dataclasses import asdict

    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
