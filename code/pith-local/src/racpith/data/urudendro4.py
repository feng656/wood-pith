from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from ..provenance import read_json_object


SAMPLE_STEM_PATTERN = re.compile(
    r"^(?P<treatment>T(?:0|2|4|6))_"
    r"(?P<block>B(?:1|2|3))_"
    r"(?P<tree_number>N\d+)_"
    r"(?P<height>ADAP|A|B|C|D)$"
)

_NUMBER_TEXT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SAMPLE_TOKEN_TEXT = (
    r"(?P<sample>T(?:0|2|4|6)_B(?:1|2|3)_N\d+_(?:ADAP|A|B|C|D)"
    r"(?:\.(?:jpg|jpeg|png|json))?)"
)
_DELIMITED_PITH_ROW = re.compile(
    rf"^\s*{_SAMPLE_TOKEN_TEXT}\s*(?P<delimiter>[,;\t])\s*"
    rf"(?P<first>{_NUMBER_TEXT})\s*(?P=delimiter)\s*"
    rf"(?P<second>{_NUMBER_TEXT})\s*$",
    re.IGNORECASE,
)
_WHITESPACE_PITH_ROW = re.compile(
    rf"^\s*{_SAMPLE_TOKEN_TEXT}[ ]+(?P<first>{_NUMBER_TEXT})[ ]+"
    rf"(?P<second>{_NUMBER_TEXT})\s*$",
    re.IGNORECASE,
)
_BRACKETED_PITH_ROW = re.compile(
    rf"^\s*{_SAMPLE_TOKEN_TEXT}\s*:\s*[\[(]\s*"
    rf"(?P<first>{_NUMBER_TEXT})\s*,\s*(?P<second>{_NUMBER_TEXT})\s*[\])]\s*$",
    re.IGNORECASE,
)

_KEY_HEADERS = {
    "code",
    "file",
    "filename",
    "image",
    "image_name",
    "sample",
    "sample_id",
    "section",
    "section_id",
    "stem",
}
_X_HEADERS = {"x", "cx", "pith_x", "pith_x_px", "x_px"}
_Y_HEADERS = {"y", "cy", "pith_y", "pith_y_px", "y_px"}


@dataclass(frozen=True)
class SampleCode:
    section_id: str
    treatment: str
    block: str
    tree_number: str
    height: str

    @property
    def tree_id(self) -> str:
        return f"{self.treatment}_{self.block}_{self.tree_number}"

    def as_dict(self) -> dict[str, str]:
        return {
            "section_id": self.section_id,
            "tree_id": self.tree_id,
            "treatment": self.treatment,
            "block": self.block,
            "tree_number": self.tree_number,
            "height": self.height,
        }


def parse_sample_code(value: str | Path) -> SampleCode:
    """Parse one official UruDendro4 sample stem without guessing.

    A suffix is removed, but directory components are forbidden.  This prevents
    digits in arbitrary paths from being mistaken for treatment/tree metadata.
    """

    text = str(value).strip()
    if not text or Path(text).name != text:
        raise ValueError(f"sample identifier must be a basename, got {value!r}")
    stem = Path(text).stem
    match = SAMPLE_STEM_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"invalid UruDendro4 sample identifier: {value!r}")
    return SampleCode(
        section_id=stem,
        treatment=match.group("treatment"),
        block=match.group("block"),
        tree_number=match.group("tree_number"),
        height=match.group("height"),
    )


def _signed_polygon_area(points_xy: np.ndarray) -> float:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_centroid(points_xy: np.ndarray) -> np.ndarray:
    cross = points_xy[:, 0] * np.roll(points_xy[:, 1], -1) - np.roll(
        points_xy[:, 0], -1
    ) * points_xy[:, 1]
    denominator = 3.0 * float(cross.sum())
    if abs(denominator) <= 1e-12:
        return points_xy.mean(axis=0)
    x = float(((points_xy[:, 0] + np.roll(points_xy[:, 0], -1)) * cross).sum())
    y = float(((points_xy[:, 1] + np.roll(points_xy[:, 1], -1)) * cross).sum())
    centroid = np.asarray([x / denominator, y / denominator], dtype=np.float64)
    if not np.all(np.isfinite(centroid)):
        return points_xy.mean(axis=0)
    return centroid


@dataclass(frozen=True)
class RingPolygon:
    ring_id: str
    ring_order: int
    source_index: int
    source_label: str
    points_xy: np.ndarray
    area_px2: float
    centroid_xy: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        points = np.ascontiguousarray(self.points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("RingPolygon.points_xy must have shape (N,2)")
        points.setflags(write=False)
        object.__setattr__(self, "points_xy", points)


@dataclass(frozen=True)
class RawPithRow:
    section_id: str
    first: float
    second: float
    line_number: int


@dataclass(frozen=True)
class RawPithTable:
    rows: tuple[RawPithRow, ...]
    dialect: str
    header_present: bool
    declared_order: str | None


@dataclass(frozen=True)
class PithResolution:
    coordinates_xy: Mapping[str, tuple[float, float]]
    selected_order: str
    order_equivalent: bool
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class UruDendro4Sample:
    code: SampleCode
    image_path: Path
    annotation_path: Path
    image_size_px: tuple[int, int]
    pith_source_px: tuple[float, float]
    rings: tuple[RingPolygon, ...]
    ring_order_reliable: bool = True

    @property
    def section_id(self) -> str:
        return self.code.section_id

    @property
    def tree_id(self) -> str:
        return self.code.tree_id

    @property
    def width(self) -> int:
        return self.image_size_px[0]

    @property
    def height(self) -> int:
        return self.image_size_px[1]

    def to_manifest_row(self) -> dict[str, Any]:
        pith_x, pith_y = self.pith_source_px
        return {
            "schema_version": "racpith.source_sample.v1",
            **self.code.as_dict(),
            "image_path": str(self.image_path.resolve()),
            "annotation_path": str(self.annotation_path.resolve()),
            # Named distinctly from the sample-stem "height" code (A/B/C/D/ADAP).
            "image_width_px": self.width,
            "image_height_px": self.height,
            "image_size_px": [self.width, self.height],
            "pith_source_x": pith_x,
            "pith_source_y": pith_y,
            "pith_source_px": [pith_x, pith_y],
            "ring_count": len(self.rings),
            "ring_order_reliable": self.ring_order_reliable,
        }


@dataclass(frozen=True)
class UruDendro4Index:
    dataset_root: Path
    samples: tuple[UruDendro4Sample, ...]
    pith_dialect: str
    pith_order: str
    pith_order_equivalent: bool
    pith_diagnostics: Mapping[str, Any]

    @property
    def by_section(self) -> dict[str, UruDendro4Sample]:
        return {sample.section_id: sample for sample in self.samples}

    @property
    def tree_ids(self) -> tuple[str, ...]:
        return tuple(sorted({sample.tree_id for sample in self.samples}))

    def manifest_rows(self) -> list[dict[str, Any]]:
        return [sample.to_manifest_row() for sample in self.samples]

    def audit_summary(self) -> dict[str, Any]:
        by_treatment: dict[str, int] = {}
        by_block: dict[str, int] = {}
        by_height: dict[str, int] = {}
        for sample in self.samples:
            by_treatment[sample.code.treatment] = by_treatment.get(sample.code.treatment, 0) + 1
            by_block[sample.code.block] = by_block.get(sample.code.block, 0) + 1
            by_height[sample.code.height] = by_height.get(sample.code.height, 0) + 1
        return {
            "schema_version": "racpith.dataset_audit.v1",
            "dataset_root": str(self.dataset_root.resolve()),
            "section_count": len(self.samples),
            "tree_count": len(self.tree_ids),
            "sections_by_treatment": dict(sorted(by_treatment.items())),
            "sections_by_block": dict(sorted(by_block.items())),
            "sections_by_height": dict(sorted(by_height.items())),
            "pith_dialect": self.pith_dialect,
            "pith_order": self.pith_order,
            "pith_order_equivalent": self.pith_order_equivalent,
            "pith_diagnostics": dict(self.pith_diagnostics),
        }


def _normalise_header(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _split_header(line: str) -> tuple[list[str], str] | None:
    candidates: list[tuple[list[str], str]] = []
    for delimiter, name in ((",", "comma"), (";", "semicolon"), ("\t", "tab")):
        fields = next(csv.reader([line], delimiter=delimiter, skipinitialspace=True))
        if len(fields) == 3:
            candidates.append((fields, name))
    if not any(delimiter in line for delimiter in (",", ";", "\t")):
        whitespace = line.split()
        if len(whitespace) == 3:
            candidates.append((whitespace, "whitespace"))
    recognised: list[tuple[list[str], str]] = []
    for fields, name in candidates:
        normalised = [_normalise_header(value) for value in fields]
        if (
            sum(value in _KEY_HEADERS for value in normalised) == 1
            and sum(value in _X_HEADERS for value in normalised) == 1
            and sum(value in _Y_HEADERS for value in normalised) == 1
        ):
            recognised.append((normalised, name))
    if len(recognised) > 1:
        raise ValueError("pith header matches more than one supported dialect")
    return recognised[0] if recognised else None


def _parse_sample_token(value: str, *, line_number: int) -> str:
    token = value.strip()
    try:
        return parse_sample_code(token).section_id
    except ValueError as exc:
        raise ValueError(
            f"pith_location line {line_number}: invalid sample identifier {token!r}"
        ) from exc


def _finite_number(value: str, *, line_number: int, column: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"pith_location line {line_number}: {column} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"pith_location line {line_number}: {column} is not finite")
    return number


def parse_pith_location(path: str | Path) -> RawPithTable:
    """Parse only documented, auditable table-like encodings.

    Headerless data deliberately retain the two numeric columns as ``first``
    and ``second``.  Their coordinate order is resolved later using image and
    ring geometry; this function never silently assumes x/y.
    """

    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    content = [
        (line_number, line.strip())
        for line_number, line in enumerate(lines, start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not content:
        raise ValueError(f"empty pith location file: {source}")

    header = _split_header(content[0][1])
    rows: list[RawPithRow] = []
    if header is not None:
        columns, dialect = header
        key_index = next(i for i, value in enumerate(columns) if value in _KEY_HEADERS)
        x_index = next(i for i, value in enumerate(columns) if value in _X_HEADERS)
        y_index = next(i for i, value in enumerate(columns) if value in _Y_HEADERS)
        delimiter = {"comma": ",", "semicolon": ";", "tab": "\t"}.get(dialect)
        for line_number, line in content[1:]:
            fields = line.split() if delimiter is None else next(
                csv.reader([line], delimiter=delimiter, skipinitialspace=True)
            )
            if len(fields) != 3:
                raise ValueError(
                    f"pith_location line {line_number}: expected exactly 3 {dialect} fields"
                )
            rows.append(
                RawPithRow(
                    section_id=_parse_sample_token(fields[key_index], line_number=line_number),
                    first=_finite_number(fields[x_index], line_number=line_number, column="x"),
                    second=_finite_number(fields[y_index], line_number=line_number, column="y"),
                    line_number=line_number,
                )
            )
        declared_order: str | None = "xy"
        header_present = True
    else:
        first_line_number, first_line = content[0]
        matches: list[tuple[str, re.Match[str]]] = []
        for name, pattern in (
            ("delimited", _DELIMITED_PITH_ROW),
            ("whitespace", _WHITESPACE_PITH_ROW),
            ("bracketed", _BRACKETED_PITH_ROW),
        ):
            match = pattern.fullmatch(first_line)
            if match is not None:
                matches.append((name, match))
        if len(matches) != 1:
            raise ValueError(
                f"pith_location line {first_line_number}: row does not match exactly one "
                "supported headerless format"
            )
        dialect, _ = matches[0]
        pattern = {
            "delimited": _DELIMITED_PITH_ROW,
            "whitespace": _WHITESPACE_PITH_ROW,
            "bracketed": _BRACKETED_PITH_ROW,
        }[dialect]
        delimiter_seen: str | None = None
        for line_number, line in content:
            match = pattern.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"pith_location line {line_number}: mixed or unsupported row format"
                )
            if dialect == "delimited":
                current = match.group("delimiter")
                if delimiter_seen is None:
                    delimiter_seen = current
                elif current != delimiter_seen:
                    raise ValueError(
                        f"pith_location line {line_number}: delimiter differs from preceding rows"
                    )
            rows.append(
                RawPithRow(
                    section_id=_parse_sample_token(match.group("sample"), line_number=line_number),
                    first=_finite_number(
                        match.group("first"), line_number=line_number, column="first coordinate"
                    ),
                    second=_finite_number(
                        match.group("second"), line_number=line_number, column="second coordinate"
                    ),
                    line_number=line_number,
                )
            )
        if dialect == "delimited":
            dialect = {",": "comma", ";": "semicolon", "\t": "tab"}[delimiter_seen]
        declared_order = None
        header_present = False

    if not rows:
        raise ValueError("pith location file contains a header but no data rows")
    seen: dict[str, int] = {}
    for row in rows:
        if row.section_id in seen:
            raise ValueError(
                f"pith_location line {row.line_number}: duplicate {row.section_id!r}; "
                f"first seen on line {seen[row.section_id]}"
            )
        seen[row.section_id] = row.line_number
    return RawPithTable(
        rows=tuple(rows),
        dialect=dialect,
        header_present=header_present,
        declared_order=declared_order,
    )


def _point_polygon_signed_distance(point_xy: Sequence[float], polygon_xy: np.ndarray) -> float:
    contour = np.asarray(polygon_xy, dtype=np.float32).reshape((-1, 1, 2))
    return float(cv2.pointPolygonTest(contour, (float(point_xy[0]), float(point_xy[1])), True))


def _candidate_pith_diagnostics(
    coordinates: Mapping[str, tuple[float, float]],
    geometry: Mapping[str, tuple[tuple[int, int], RingPolygon]],
    *,
    containment_tolerance_px: float,
) -> dict[str, Any]:
    bounds_failures: list[str] = []
    containment_failures: list[str] = []
    normalised_centroid_distances: list[float] = []
    minimum_boundary_distances: list[float] = []
    for section_id in sorted(geometry):
        width_height, inner_ring = geometry[section_id]
        width, height = width_height
        x, y = coordinates[section_id]
        if not (0.0 <= x < float(width) and 0.0 <= y < float(height)):
            bounds_failures.append(section_id)
            continue
        signed_distance = _point_polygon_signed_distance((x, y), inner_ring.points_xy)
        minimum_boundary_distances.append(signed_distance)
        if signed_distance < -containment_tolerance_px:
            containment_failures.append(section_id)
        radius = math.sqrt(inner_ring.area_px2 / math.pi)
        normalised_centroid_distances.append(
            math.hypot(x - inner_ring.centroid_xy[0], y - inner_ring.centroid_xy[1])
            / max(radius, 1e-12)
        )
    return {
        "bounds_failure_count": len(bounds_failures),
        "bounds_failure_examples": bounds_failures[:10],
        "innermost_containment_failure_count": len(containment_failures),
        "innermost_containment_failure_examples": containment_failures[:10],
        "median_distance_to_innermost_centroid_over_equivalent_radius": (
            float(np.median(normalised_centroid_distances))
            if normalised_centroid_distances
            else None
        ),
        "minimum_signed_distance_to_innermost_ring_px": (
            float(min(minimum_boundary_distances)) if minimum_boundary_distances else None
        ),
    }


def resolve_pith_coordinates(
    table: RawPithTable,
    geometry: Mapping[str, tuple[tuple[int, int], RingPolygon]],
    *,
    order: str = "auto",
    containment_tolerance_px: float = 1.0,
    auto_distance_margin: float = 0.05,
) -> PithResolution:
    """Resolve pith column order with bounds and innermost-ring geometry.

    ``auto`` is intentionally fail-closed.  When both orientations remain
    plausible and their robust geometric scores are too close, the caller must
    explicitly select ``xy`` or ``yx`` and the ambiguity remains auditable.
    """

    if order not in {"auto", "xy", "yx"}:
        raise ValueError("pith order must be one of: auto, xy, yx")
    row_ids = {row.section_id for row in table.rows}
    geometry_ids = set(geometry)
    if row_ids != geometry_ids:
        missing = sorted(geometry_ids - row_ids)
        extra = sorted(row_ids - geometry_ids)
        raise ValueError(
            f"pith/image stem mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    if table.declared_order == "xy" and order == "yx":
        raise ValueError("pith header explicitly names x/y columns but --pith-order=yx was requested")

    xy = {row.section_id: (row.first, row.second) for row in table.rows}
    yx = {row.section_id: (row.second, row.first) for row in table.rows}
    candidates = {"xy": xy, "yx": yx}
    diagnostics = {
        name: _candidate_pith_diagnostics(
            values, geometry, containment_tolerance_px=containment_tolerance_px
        )
        for name, values in candidates.items()
    }

    def valid(name: str) -> bool:
        values = diagnostics[name]
        return (
            values["bounds_failure_count"] == 0
            and values["innermost_containment_failure_count"] == 0
        )

    if table.declared_order == "xy":
        selected = "xy"
    elif order in {"xy", "yx"}:
        selected = order
    else:
        arrays_equal = all(
            np.allclose(xy[key], yx[key], rtol=0.0, atol=1e-12) for key in sorted(xy)
        )
        if arrays_equal:
            selected = "xy"
        else:
            plausible = [name for name in ("xy", "yx") if valid(name)]
            if len(plausible) == 1:
                selected = plausible[0]
            elif len(plausible) == 0:
                raise ValueError(
                    "neither pith coordinate order passes image-bound and innermost-ring checks; "
                    f"diagnostics={diagnostics}"
                )
            else:
                score_xy = diagnostics["xy"][
                    "median_distance_to_innermost_centroid_over_equivalent_radius"
                ]
                score_yx = diagnostics["yx"][
                    "median_distance_to_innermost_centroid_over_equivalent_radius"
                ]
                if score_xy is None or score_yx is None:
                    raise ValueError("cannot score pith coordinate order without ring geometry")
                if abs(float(score_xy) - float(score_yx)) <= auto_distance_margin:
                    raise ValueError(
                        "pith coordinate order remains ambiguous after bounds, containment, and "
                        f"distance checks; pass --pith-order explicitly; diagnostics={diagnostics}"
                    )
                selected = "xy" if float(score_xy) < float(score_yx) else "yx"

    if not valid(selected):
        raise ValueError(
            f"selected pith coordinate order {selected!r} fails geometric validation; "
            f"diagnostics={diagnostics[selected]}"
        )
    equivalent = all(
        np.allclose(xy[key], yx[key], rtol=0.0, atol=1e-12) for key in sorted(xy)
    )
    return PithResolution(
        coordinates_xy=candidates[selected],
        selected_order=selected,
        order_equivalent=equivalent,
        diagnostics=diagnostics,
    )


def _validate_labelme_points(
    value: Any,
    *,
    annotation_path: Path,
    shape_index: int,
    width: int,
    height: int,
) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{annotation_path}: shape {shape_index} points are not a numeric array"
        ) from exc
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"{annotation_path}: shape {shape_index} needs at least three [x,y] points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{annotation_path}: shape {shape_index} contains NaN or infinity")
    if len(points) > 3 and np.linalg.norm(points[0] - points[-1]) <= 1e-9:
        points = points[:-1]
    # Some official annotations contain redundant consecutive vertices (accidental
    # double clicks); collapse those zero-length segments instead of rejecting the shape.
    if len(points) > 1:
        keep = np.concatenate(
            ([True], np.linalg.norm(points[1:] - points[:-1], axis=1) > 1e-12)
        )
        points = points[keep]
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1e-12:
        points = points[:-1]
    if len(points) < 3 or len(np.unique(points, axis=0)) < 3:
        raise ValueError(f"{annotation_path}: shape {shape_index} is geometrically degenerate")
    adjacent = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
    if np.any(adjacent <= 1e-12):
        raise ValueError(f"{annotation_path}: shape {shape_index} has duplicate adjacent vertices")
    x, y = points[:, 0], points[:, 1]
    if np.any(x < 0.0) or np.any(x >= width) or np.any(y < 0.0) or np.any(y >= height):
        raise ValueError(
            f"{annotation_path}: shape {shape_index} has [x,y] outside image bounds {width}x{height}"
        )
    return points


def _optional_labelme_dimension(
    value: Any,
    *,
    annotation_path: Path,
    field: str,
) -> int | None:
    """Parse optional LabelMe size metadata without trusting blank placeholders.

    Some official UruDendro4 annotations retain ``imageWidth`` and
    ``imageHeight`` but store them as empty strings.  Those values carry no
    size information, so the decoded source image remains authoritative.  A
    non-empty declaration is still validated strictly.
    """

    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    if isinstance(value, bool):
        raise ValueError(
            f"{annotation_path}: Labelme {field} must be an integer or blank"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{annotation_path}: Labelme {field} must be an integer or blank, got {value!r}"
        ) from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(
            f"{annotation_path}: Labelme {field} must be a positive integer or blank, "
            f"got {value!r}"
        )
    return int(numeric)


def _containment_probes(points_xy: np.ndarray, maximum: int = 512) -> np.ndarray:
    if len(points_xy) <= maximum // 2:
        midpoints = 0.5 * (points_xy + np.roll(points_xy, -1, axis=0))
        return np.vstack([points_xy, midpoints])
    indices = np.linspace(0, len(points_xy) - 1, maximum // 2, dtype=np.int64)
    selected = points_xy[indices]
    successors = points_xy[(indices + 1) % len(points_xy)]
    return np.vstack([selected, 0.5 * (selected + successors)])


def _nested_inside(inner: RingPolygon, outer: RingPolygon, tolerance_px: float) -> bool:
    ix0, iy0, ix1, iy1 = inner.bbox_xyxy
    ox0, oy0, ox1, oy1 = outer.bbox_xyxy
    if (
        ix0 < ox0 - tolerance_px
        or iy0 < oy0 - tolerance_px
        or ix1 > ox1 + tolerance_px
        or iy1 > oy1 + tolerance_px
    ):
        return False
    contour = outer.points_xy.astype(np.float32).reshape((-1, 1, 2))
    for point in _containment_probes(inner.points_xy):
        if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True) < -tolerance_px:
            return False
    return True


def load_labelme_rings(
    annotation_path: str | Path,
    *,
    expected_section_id: str,
    expected_image_size_px: tuple[int, int],
    nesting_tolerance_px: float = 1.0,
) -> tuple[RingPolygon, ...]:
    """Load Labelme polygons and infer inner-to-outer order from geometry only."""

    path = Path(annotation_path)
    try:
        payload = read_json_object(path, encoding="utf-8-sig")
    except ValueError as exc:
        raise ValueError(f"invalid Labelme JSON {path}: {exc}") from exc
    required = {
        "version",
        "flags",
        "shapes",
        "imageData",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path}: missing Labelme fields {missing}")
    if not isinstance(payload["flags"], dict):
        raise ValueError(f"{path}: top-level Labelme flags must be an object")
    width, height = expected_image_size_px
    declared_width = _optional_labelme_dimension(
        payload.get("imageWidth"),
        annotation_path=path,
        field="imageWidth",
    )
    declared_height = _optional_labelme_dimension(
        payload.get("imageHeight"),
        annotation_path=path,
        field="imageHeight",
    )
    width_mismatch = declared_width is not None and declared_width != width
    height_mismatch = declared_height is not None and declared_height != height
    if width_mismatch or height_mismatch:
        raise ValueError(
            f"{path}: Labelme declared dimensions {(declared_width, declared_height)} "
            f"do not match image {(width, height)}"
        )
    image_path_value = payload.get("imagePath")
    if image_path_value is not None and not isinstance(image_path_value, str):
        raise ValueError(
            f"{path}: Labelme imagePath must be a string, null, or absent"
        )
    if isinstance(image_path_value, str) and image_path_value.strip():
        declared_stem = Path(image_path_value.strip().replace("\\", "/")).stem
        if declared_stem != expected_section_id:
            raise ValueError(
                f"{path}: Labelme imagePath does not match section {expected_section_id}"
            )
    shapes = payload["shapes"]
    if not isinstance(shapes, list) or not shapes:
        raise ValueError(f"{path}: Labelme shapes must be a non-empty list")

    unordered: list[RingPolygon] = []
    for source_index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise ValueError(f"{path}: shape {source_index} is not an object")
        shape_required = {"label", "points", "shape_type", "flags"}
        shape_missing = sorted(shape_required - set(shape))
        if shape_missing:
            raise ValueError(f"{path}: shape {source_index} missing fields {shape_missing}")
        if shape["shape_type"] != "polygon":
            raise ValueError(
                f"{path}: shape {source_index} has shape_type={shape['shape_type']!r}, expected polygon"
            )
        if not isinstance(shape["flags"], dict):
            raise ValueError(f"{path}: shape {source_index} flags must be an object")
        points = _validate_labelme_points(
            shape["points"],
            annotation_path=path,
            shape_index=source_index,
            width=width,
            height=height,
        )
        area = abs(_signed_polygon_area(points))
        if area <= 1e-6:
            raise ValueError(f"{path}: shape {source_index} has zero polygon area")
        centroid = _polygon_centroid(points)
        unordered.append(
            RingPolygon(
                ring_id="",
                ring_order=-1,
                source_index=source_index,
                source_label=str(shape["label"]),
                points_xy=points,
                area_px2=area,
                centroid_xy=(float(centroid[0]), float(centroid[1])),
                bbox_xyxy=(
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max()),
                    float(points[:, 1].max()),
                ),
            )
        )
    unordered.sort(key=lambda ring: (ring.area_px2, ring.source_index))
    for inner, outer in zip(unordered, unordered[1:]):
        if outer.area_px2 - inner.area_px2 <= 1e-6:
            raise ValueError(f"{path}: ring areas are tied; geometric order is not unique")
        if not _nested_inside(inner, outer, nesting_tolerance_px):
            raise ValueError(
                f"{path}: polygons {inner.source_index} and {outer.source_index} are not nested; "
                "ring order is not reliable"
            )
    return tuple(
        RingPolygon(
            ring_id=f"ring_{order:03d}",
            ring_order=order,
            source_index=ring.source_index,
            source_label=ring.source_label,
            points_xy=ring.points_xy,
            area_px2=ring.area_px2,
            centroid_xy=ring.centroid_xy,
            bbox_xyxy=ring.bbox_xyxy,
        )
        for order, ring in enumerate(unordered)
    )


def _resolve_dataset_child(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ValueError(f"dataset child path must be relative, got {relative!r}")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"dataset child path escapes root: {relative!r}")
    return resolved


def _index_flat_files(directory: Path, allowed_suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.name.startswith("."):
            continue
        if not entry.is_file():
            raise ValueError(f"unexpected non-file entry in flat dataset directory: {entry}")
        if entry.suffix.lower() not in allowed_suffixes:
            raise ValueError(f"unexpected file suffix in {directory}: {entry.name}")
        code = parse_sample_code(entry.name)
        if code.section_id in result:
            raise ValueError(f"duplicate sample stem {code.section_id!r} in {directory}")
        result[code.section_id] = entry.resolve()
    if not result:
        raise ValueError(f"dataset directory has no supported files: {directory}")
    return result


def _validate_official_complete(samples: Sequence[UruDendro4Sample]) -> None:
    if len(samples) != 102:
        raise ValueError(f"official UruDendro4 v1 must have 102 sections, found {len(samples)}")
    groups: dict[str, list[UruDendro4Sample]] = {}
    cells: dict[tuple[str, str], set[str]] = {}
    for sample in samples:
        groups.setdefault(sample.tree_id, []).append(sample)
        cells.setdefault((sample.code.treatment, sample.code.block), set()).add(sample.tree_id)
    if len(groups) != 24:
        raise ValueError(f"official UruDendro4 v1 must have 24 trees, found {len(groups)}")
    expected_cells = {(f"T{t}", f"B{b}") for t in (0, 2, 4, 6) for b in (1, 2, 3)}
    if set(cells) != expected_cells or any(len(values) != 2 for values in cells.values()):
        raise ValueError("official UruDendro4 v1 must contain two trees in every treatment/block cell")
    for tree_id, members in groups.items():
        treatment = members[0].code.treatment
        expected_heights = {"A", "B", "C", "D", "ADAP"} if treatment == "T0" else {
            "A",
            "B",
            "C",
            "D",
        }
        actual_heights = {member.code.height for member in members}
        if actual_heights != expected_heights or len(members) != len(expected_heights):
            raise ValueError(
                f"official UruDendro4 tree {tree_id} has heights {sorted(actual_heights)}, "
                f"expected {sorted(expected_heights)}"
            )


def load_urudendro4(
    dataset_root: str | Path,
    *,
    image_dir: str | Path = "images_no_background",
    annotation_dir: str | Path = "annotations/annual_rings",
    pith_file: str | Path = "pith_location.txt",
    pith_order: str = "auto",
    require_official_complete: bool = True,
) -> UruDendro4Index:
    """Build a read-only, strict stem-joined UruDendro4 index."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    image_paths = _index_flat_files(
        _resolve_dataset_child(root, image_dir), {".jpg", ".jpeg", ".png"}
    )
    annotation_paths = _index_flat_files(_resolve_dataset_child(root, annotation_dir), {".json"})
    if set(image_paths) != set(annotation_paths):
        missing_annotations = sorted(set(image_paths) - set(annotation_paths))
        missing_images = sorted(set(annotation_paths) - set(image_paths))
        raise ValueError(
            "image/annotation stem mismatch; "
            f"missing_annotations={missing_annotations[:10]}, missing_images={missing_images[:10]}"
        )

    intermediate: dict[
        str, tuple[SampleCode, Path, Path, tuple[int, int], tuple[RingPolygon, ...]]
    ] = {}
    geometry: dict[str, tuple[tuple[int, int], RingPolygon]] = {}
    for section_id in sorted(image_paths):
        image_path = image_paths[section_id]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim not in {2, 3}:
            raise ValueError(f"cannot decode image or unsupported image rank: {image_path}")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError(f"image has invalid dimensions: {image_path}")
        size = (int(width), int(height))
        rings = load_labelme_rings(
            annotation_paths[section_id],
            expected_section_id=section_id,
            expected_image_size_px=size,
        )
        code = parse_sample_code(section_id)
        intermediate[section_id] = (
            code,
            image_path,
            annotation_paths[section_id],
            size,
            rings,
        )
        geometry[section_id] = (size, rings[0])

    raw_pith = parse_pith_location(_resolve_dataset_child(root, pith_file))
    resolution = resolve_pith_coordinates(raw_pith, geometry, order=pith_order)
    samples = tuple(
        UruDendro4Sample(
            code=intermediate[section_id][0],
            image_path=intermediate[section_id][1],
            annotation_path=intermediate[section_id][2],
            image_size_px=intermediate[section_id][3],
            pith_source_px=resolution.coordinates_xy[section_id],
            rings=intermediate[section_id][4],
            ring_order_reliable=True,
        )
        for section_id in sorted(intermediate)
    )
    if require_official_complete:
        _validate_official_complete(samples)
    return UruDendro4Index(
        dataset_root=root,
        samples=samples,
        pith_dialect=raw_pith.dialect,
        pith_order=resolution.selected_order,
        pith_order_equivalent=resolution.order_equivalent,
        pith_diagnostics=resolution.diagnostics,
    )


def source_manifest_rows(index: UruDendro4Index) -> Iterable[dict[str, Any]]:
    """Yield deterministic source rows without writing into the dataset."""

    yield from index.manifest_rows()
