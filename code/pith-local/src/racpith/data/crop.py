from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .urudendro4 import RingPolygon, UruDendro4Sample


@dataclass(frozen=True, order=True)
class CropBox:
    """Integer source-image rectangle with half-open bounds [x0,x1) x [y0,y1)."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"invalid half-open crop box: {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def contains_xy(self, point_xy: Sequence[float]) -> bool:
        x, y = float(point_xy[0]), float(point_xy[1])
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def crop_pith_coordinates(
    pith_source_px: Sequence[float], box: CropBox
) -> tuple[tuple[float, float], bool]:
    """Translate one source pith coordinate into a crop-local coordinate."""

    if len(pith_source_px) != 2:
        raise ValueError("pith_source_px must contain exactly two coordinates")
    source_x, source_y = (float(value) for value in pith_source_px)
    if not math.isfinite(source_x) or not math.isfinite(source_y):
        raise ValueError("pith_source_px must be finite")
    pith_crop_px = (source_x - box.x0, source_y - box.y0)
    return pith_crop_px, box.contains_xy((source_x, source_y))


@dataclass(frozen=True)
class CropRecord:
    crop_id: str
    tree_id: str
    section_id: str
    split: str
    box: CropBox
    scale_index: int
    scale_fraction: float
    eligible: bool
    selected: bool
    rejection_reasons: tuple[str, ...]
    foreground_fraction: float
    visible_parent_rings: int
    distance_value: float
    distance_stratum: str
    pith_source_px: tuple[float, float]
    pith_crop_px: tuple[float, float]
    pith_inside_crop: bool
    source_image_path: Path
    source_annotation_path: Path
    crop_image_path: Path
    crop_annotation_path: Path

    @property
    def crop_size_px(self) -> tuple[int, int]:
        return self.box.width, self.box.height

    @property
    def normalization_scale_px(self) -> float:
        return float(math.hypot(self.box.width, self.box.height))

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "schema_version": "racpith.crop_manifest.v1",
            "crop_id": self.crop_id,
            "tree_id": self.tree_id,
            "section_id": self.section_id,
            "split": self.split,
            "eligible": self.eligible,
            "selected": self.selected,
            "rejection_reasons": list(self.rejection_reasons),
            "source_image_path": str(self.source_image_path.resolve()),
            "source_annotation_path": str(self.source_annotation_path.resolve()),
            "crop_image_path": str(self.crop_image_path.resolve()),
            "crop_annotation_path": str(self.crop_annotation_path.resolve()),
            "crop_box_source_px": self.box.as_list(),
            "crop_origin_source_px": [self.box.x0, self.box.y0],
            "crop_size_px": [self.box.width, self.box.height],
            "normalization_scale_px": self.normalization_scale_px,
            "scale_index": self.scale_index,
            "scale_fraction_of_min_side": self.scale_fraction,
            "foreground_fraction": self.foreground_fraction,
            "visible_parent_rings": self.visible_parent_rings,
            "pith_source_px": list(self.pith_source_px),
            "pith_crop_px": list(self.pith_crop_px),
            "pith_inside_crop": self.pith_inside_crop,
            "distance_value": self.distance_value,
            "distance_stratum": self.distance_stratum,
        }


@dataclass(frozen=True)
class PreparedCrop:
    record: CropRecord
    annotation: Mapping[str, Any]

    def crop_image(self, source_image: np.ndarray) -> np.ndarray:
        box = self.record.box
        if source_image.shape[1] < box.x1 or source_image.shape[0] < box.y1:
            raise ValueError("source image is smaller than declared crop box")
        return source_image[box.y0 : box.y1, box.x0 : box.x1].copy()


@dataclass(frozen=True)
class SectionCropPlan:
    section_id: str
    tree_id: str
    split: str
    candidates: tuple[PreparedCrop, ...]

    @property
    def selected(self) -> tuple[PreparedCrop, ...]:
        return tuple(item for item in self.candidates if item.record.selected)

    def manifest_rows(self, *, selected_only: bool = True) -> list[dict[str, Any]]:
        values = self.selected if selected_only else self.candidates
        return [item.record.to_manifest_row() for item in values]

    def audit_summary(self) -> dict[str, Any]:
        strata: dict[str, int] = {}
        rejection_reasons: dict[str, int] = {}
        for item in self.candidates:
            if item.record.selected:
                strata[item.record.distance_stratum] = strata.get(item.record.distance_stratum, 0) + 1
            for reason in item.record.rejection_reasons:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        return {
            "schema_version": "racpith.crop_plan_audit.v1",
            "section_id": self.section_id,
            "tree_id": self.tree_id,
            "split": self.split,
            "candidate_count": len(self.candidates),
            "eligible_count": sum(item.record.eligible for item in self.candidates),
            "selected_count": len(self.selected),
            "selected_by_distance_stratum": dict(sorted(strata.items())),
            "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        }


@dataclass
class _SegmentPiece:
    segment_index: int
    t0: float
    t1: float
    source_s0: float
    source_s1: float
    points_source: np.ndarray
    source_s: np.ndarray

    def lineage(self, *, unwrapped_shift: float = 0.0) -> dict[str, Any]:
        return {
            "source_segment_index": self.segment_index,
            "source_t_interval": [self.t0, self.t1],
            "source_s_interval_px": [self.source_s0, self.source_s1],
            "unwrapped_source_s_interval_px": [
                self.source_s0 + unwrapped_shift,
                self.source_s1 + unwrapped_shift,
            ],
        }


@dataclass
class _Fragment:
    pieces: list[_SegmentPiece]
    points_source: np.ndarray
    source_s: np.ndarray
    wraps_source_seam: bool = False
    first_piece_shift: int = 0


def _axis_starts(length: int, crop_size: int, stride: int) -> list[int]:
    if crop_size > length:
        raise ValueError("crop size cannot exceed image dimension")
    if crop_size == length:
        return [0]
    starts = list(range(0, length - crop_size + 1, stride))
    final = length - crop_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def overlapping_crop_boxes(
    image_size_px: tuple[int, int],
    size_fractions_of_min_side: Sequence[float],
    *,
    stride_fraction_of_crop: float,
) -> list[tuple[int, float, CropBox]]:
    width, height = image_size_px
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < stride_fraction_of_crop < 1.0:
        raise ValueError("stride_fraction_of_crop must be in (0,1) to produce overlapping crops")
    if not size_fractions_of_min_side:
        raise ValueError("at least one crop scale is required")
    minimum_side = min(width, height)
    realised_sizes: set[int] = set()
    result: list[tuple[int, float, CropBox]] = []
    for scale_index, raw_fraction in enumerate(size_fractions_of_min_side):
        fraction = float(raw_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError(f"invalid crop size fraction {raw_fraction!r}")
        crop_size = max(2, int(round(fraction * minimum_side)))
        if crop_size in realised_sizes:
            raise ValueError(
                f"crop fractions produce duplicate realised size {crop_size}px; configuration is ambiguous"
            )
        realised_sizes.add(crop_size)
        stride = max(1, int(round(stride_fraction_of_crop * crop_size)))
        for y0 in _axis_starts(height, crop_size, stride):
            for x0 in _axis_starts(width, crop_size, stride):
                result.append(
                    (
                        scale_index,
                        fraction,
                        CropBox(x0=x0, y0=y0, x1=x0 + crop_size, y1=y0 + crop_size),
                    )
                )
    return result


def foreground_mask(image: np.ndarray, *, threshold: int = 3) -> np.ndarray:
    if image.ndim == 2:
        return image > threshold
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("source image must be grayscale, BGR/RGB, or BGRA/RGBA")
    if image.shape[2] == 4:
        return image[:, :, 3] > threshold
    return np.max(image[:, :, :3], axis=2) > threshold


def _clip_segment_half_open(
    p0: np.ndarray, p1: np.ndarray, box: CropBox
) -> tuple[float, float] | None:
    """Liang-Barsky clip against a half-open box, excluding right/bottom exactly."""

    xmin = float(box.x0)
    ymin = float(box.y0)
    xmax = float(np.nextafter(float(box.x1), -math.inf))
    ymax = float(np.nextafter(float(box.y1), -math.inf))
    delta = p1 - p0
    t_enter, t_exit = 0.0, 1.0
    constraints = (
        (-float(delta[0]), float(p0[0]) - xmin),
        (float(delta[0]), xmax - float(p0[0])),
        (-float(delta[1]), float(p0[1]) - ymin),
        (float(delta[1]), ymax - float(p0[1])),
    )
    for coefficient, offset in constraints:
        if abs(coefficient) <= 1e-15:
            if offset < 0.0:
                return None
            continue
        ratio = offset / coefficient
        if coefficient < 0.0:
            t_enter = max(t_enter, ratio)
        else:
            t_exit = min(t_exit, ratio)
        if t_enter > t_exit:
            return None
    t_enter = min(max(t_enter, 0.0), 1.0)
    t_exit = min(max(t_exit, 0.0), 1.0)
    if t_exit - t_enter <= 1e-12:
        return None
    return t_enter, t_exit


def _segment_piece(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    segment_index: int,
    segment_start_s: float,
    segment_length: float,
    t0: float,
    t1: float,
    maximum_vertex_spacing_px: float,
) -> _SegmentPiece:
    visible_length = segment_length * (t1 - t0)
    interval_count = max(1, int(math.ceil(visible_length / maximum_vertex_spacing_px)))
    parameters = np.linspace(t0, t1, interval_count + 1, dtype=np.float64)
    points = p0[None, :] + parameters[:, None] * (p1 - p0)[None, :]
    source_s = segment_start_s + parameters * segment_length
    return _SegmentPiece(
        segment_index=segment_index,
        t0=float(t0),
        t1=float(t1),
        source_s0=float(source_s[0]),
        source_s1=float(source_s[-1]),
        points_source=points,
        source_s=source_s,
    )


def _pieces_are_contiguous(left: _SegmentPiece, right: _SegmentPiece) -> bool:
    return (
        right.segment_index == left.segment_index + 1
        and abs(left.t1 - 1.0) <= 1e-10
        and abs(right.t0) <= 1e-10
        and np.linalg.norm(left.points_source[-1] - right.points_source[0]) <= 1e-7
    )


def _fragment_from_pieces(pieces: list[_SegmentPiece]) -> _Fragment:
    points = pieces[0].points_source.copy()
    source_s = pieces[0].source_s.copy()
    for piece in pieces[1:]:
        points = np.vstack([points, piece.points_source[1:]])
        source_s = np.concatenate([source_s, piece.source_s[1:]])
    return _Fragment(pieces=list(pieces), points_source=points, source_s=source_s)


def _clip_ring_fragments(
    ring: RingPolygon,
    box: CropBox,
    *,
    maximum_vertex_spacing_px: float,
) -> list[dict[str, Any]]:
    points = ring.points_xy
    successors = np.roll(points, -1, axis=0)
    segment_lengths = np.linalg.norm(successors - points, axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise ValueError(f"ring {ring.ring_id} has a zero-length source segment")
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    total_length = float(cumulative[-1])

    pieces: list[_SegmentPiece] = []
    for index, (p0, p1, length) in enumerate(zip(points, successors, segment_lengths)):
        clipped = _clip_segment_half_open(p0, p1, box)
        if clipped is None:
            continue
        pieces.append(
            _segment_piece(
                p0,
                p1,
                segment_index=index,
                segment_start_s=float(cumulative[index]),
                segment_length=float(length),
                t0=clipped[0],
                t1=clipped[1],
                maximum_vertex_spacing_px=maximum_vertex_spacing_px,
            )
        )
    if not pieces:
        return []

    groups: list[list[_SegmentPiece]] = [[pieces[0]]]
    for piece in pieces[1:]:
        if _pieces_are_contiguous(groups[-1][-1], piece):
            groups[-1].append(piece)
        else:
            groups.append([piece])
    fragments = [_fragment_from_pieces(group) for group in groups]

    if len(fragments) > 1:
        first = fragments[0]
        last = fragments[-1]
        source_seam_contiguous = (
            first.pieces[0].segment_index == 0
            and abs(first.pieces[0].t0) <= 1e-10
            and last.pieces[-1].segment_index == len(points) - 1
            and abs(last.pieces[-1].t1 - 1.0) <= 1e-10
            and np.linalg.norm(last.points_source[-1] - first.points_source[0]) <= 1e-7
        )
        if source_seam_contiguous:
            shifted_first_s = first.source_s + total_length
            merged = _Fragment(
                pieces=last.pieces + first.pieces,
                points_source=np.vstack([last.points_source, first.points_source[1:]]),
                source_s=np.concatenate([last.source_s, shifted_first_s[1:]]),
                wraps_source_seam=True,
                first_piece_shift=len(last.pieces),
            )
            fragments = fragments[1:-1] + [merged]

    output: list[dict[str, Any]] = []
    for fragment in fragments:
        fully_closed = (
            len(fragment.pieces) == len(points)
            and fragment.pieces[0].segment_index == 0
            and abs(fragment.pieces[0].t0) <= 1e-10
            and fragment.pieces[-1].segment_index == len(points) - 1
            and abs(fragment.pieces[-1].t1 - 1.0) <= 1e-10
            and abs(sum(piece.source_s1 - piece.source_s0 for piece in fragment.pieces) - total_length)
            <= max(1e-7, total_length * 1e-10)
        )
        points_source = fragment.points_source
        source_s = fragment.source_s
        if fully_closed and np.linalg.norm(points_source[0] - points_source[-1]) <= 1e-7:
            points_source = points_source[:-1]
            source_s = source_s[:-1]
        points_crop = points_source - np.asarray([box.x0, box.y0], dtype=np.float64)
        if len(points_crop) < 2:
            continue
        visible_length = float(np.linalg.norm(np.diff(points_source, axis=0), axis=1).sum())
        if fully_closed:
            visible_length += float(np.linalg.norm(points_source[0] - points_source[-1]))

        lineage: list[dict[str, Any]] = []
        for piece_index, piece in enumerate(fragment.pieces):
            shift = (
                total_length
                if fragment.wraps_source_seam and piece_index >= fragment.first_piece_shift
                else 0.0
            )
            lineage.append(piece.lineage(unwrapped_shift=shift))
        source_interval = (
            [0.0, total_length]
            if fully_closed
            else [float(source_s[0]), float(source_s[-1])]
        )
        output.append(
            {
                "points_crop_px": points_crop.tolist(),
                "points_source_px": points_source.tolist(),
                "source_s_px": source_s.tolist(),
                "closed": fully_closed,
                "visible_length_px": visible_length,
                "fragment_lineage": {
                    "source_ring_id": ring.ring_id,
                    "source_ring_length_px": total_length,
                    "source_s_interval_px": source_interval,
                    "wraps_source_seam": fragment.wraps_source_seam,
                    "source_segment_spans": lineage,
                    "gap_before": "none" if fully_closed else "crop_boundary",
                    "gap_after": "none" if fully_closed else "crop_boundary",
                },
            }
        )
    output.sort(
        key=lambda value: (
            float(value["fragment_lineage"]["source_s_interval_px"][0]) % total_length,
            float(value["fragment_lineage"]["source_s_interval_px"][1]),
        )
    )
    return output


def _signed_distance_to_crop(point_xy: Sequence[float], box: CropBox) -> float:
    x, y = float(point_xy[0]), float(point_xy[1])
    diagonal = max(math.hypot(box.width, box.height), 1e-12)
    if box.contains_xy((x, y)):
        inset = min(x - box.x0, box.x1 - x, y - box.y0, box.y1 - y)
        # Keep a point on the inclusive left/top edge in the configured
        # ``inside`` negative interval instead of aliasing it to outside zero.
        return -float(max(inset / diagonal, np.finfo(np.float64).eps))
    upper_x = float(np.nextafter(float(box.x1), -math.inf))
    upper_y = float(np.nextafter(float(box.y1), -math.inf))
    dx = max(float(box.x0) - x, 0.0, x - upper_x)
    dy = max(float(box.y0) - y, 0.0, y - upper_y)
    return float(math.hypot(dx, dy) / diagonal)


def _validate_distance_strata(
    values: Sequence[Sequence[Any]],
) -> tuple[tuple[str, float, float], ...]:
    result: list[tuple[str, float, float]] = []
    names: set[str] = set()
    for raw in values:
        if len(raw) != 3:
            raise ValueError("each distance stratum must be [name, lower, upper]")
        name, lower, upper = str(raw[0]), float(raw[1]), float(raw[2])
        if not name or name in names:
            raise ValueError(f"duplicate or empty distance stratum name: {name!r}")
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError(f"invalid distance interval for stratum {name!r}")
        names.add(name)
        result.append((name, lower, upper))
    ordered = sorted(result, key=lambda item: item[1])
    for left, right in zip(ordered, ordered[1:]):
        if right[1] < left[2] - 1e-12:
            raise ValueError(f"overlapping distance strata {left[0]!r} and {right[0]!r}")
    return tuple(ordered)


def _distance_stratum(
    distance_value: float, strata: Sequence[tuple[str, float, float]]
) -> str | None:
    matches = [
        name
        for index, (name, lower, upper) in enumerate(strata)
        if lower <= distance_value < upper
        or (index == len(strata) - 1 and distance_value == upper)
    ]
    if len(matches) > 1:
        raise ValueError("distance value belongs to overlapping strata")
    return matches[0] if matches else None


def build_crop_annotation(
    sample: UruDendro4Sample,
    box: CropBox,
    *,
    crop_id: str,
    split: str,
    minimum_fragment_length_px: float = 12.0,
    maximum_vertex_spacing_px: float = 4.0,
) -> dict[str, Any]:
    """Clip source ring polylines without adding artificial rectangle edges."""

    if box.x1 > sample.width or box.y1 > sample.height:
        raise ValueError("crop box extends beyond source image")
    if minimum_fragment_length_px <= 0.0 or maximum_vertex_spacing_px <= 0.0:
        raise ValueError("fragment length and vertex spacing must be positive")
    rings_payload: list[dict[str, Any]] = []
    visible_parent_rings = 0
    total_fragment_count = 0
    short_fragment_count = 0
    for ring in sample.rings:
        fragments = _clip_ring_fragments(
            ring, box, maximum_vertex_spacing_px=maximum_vertex_spacing_px
        )
        if not fragments:
            continue
        qualifying = 0
        arcs: list[dict[str, Any]] = []
        for fragment_index, fragment in enumerate(fragments):
            visible_length = float(fragment["visible_length_px"])
            is_qualifying = visible_length >= minimum_fragment_length_px
            qualifying += int(is_qualifying)
            short_fragment_count += int(not is_qualifying)
            total_fragment_count += 1
            arcs.append(
                {
                    "ring_id": ring.ring_id,
                    "ring_order": ring.ring_order,
                    "arc_id": f"{crop_id}:{ring.ring_id}:fragment_{fragment_index:03d}",
                    "fragment_index": fragment_index,
                    "qualifies_for_evidence": is_qualifying,
                    **fragment,
                }
            )
        visible_parent_rings += int(qualifying > 0)
        rings_payload.append(
            {
                "ring_id": ring.ring_id,
                "ring_order": ring.ring_order,
                "source_shape_index": ring.source_index,
                "source_label": ring.source_label,
                "source_area_px2": ring.area_px2,
                "arcs": arcs,
            }
        )
    return {
        "schema_version": "racpith.crop_annotation.v1",
        "crop_id": crop_id,
        "tree_id": sample.tree_id,
        "section_id": sample.section_id,
        "split": split,
        "source_image_path": str(sample.image_path.resolve()),
        "source_annotation_path": str(sample.annotation_path.resolve()),
        "source_image_size_px": [sample.width, sample.height],
        "crop_box_source_px": box.as_list(),
        "crop_origin_source_px": [box.x0, box.y0],
        "crop_size_px": [box.width, box.height],
        "normalization_scale_px": float(math.hypot(box.width, box.height)),
        "visible_parent_rings": visible_parent_rings,
        "rings": rings_payload,
        "metadata": {
            "coordinate_convention": "Cartesian [x,y]; crop box is half-open",
            "ring_order": "inner_to_outer_zero_based",
            "ring_order_source": "polygon_area_plus_nested_geometry",
            "ring_order_reliable": sample.ring_order_reliable,
            "curve_crop_operation": "source_polyline_segment_intersection",
            "artificial_crop_boundary_edges_added": False,
            "minimum_fragment_length_px": minimum_fragment_length_px,
            "maximum_vertex_spacing_px": maximum_vertex_spacing_px,
            "total_fragment_count": total_fragment_count,
            "short_fragment_count": short_fragment_count,
        },
    }


def _safe_component(value: str, *, field: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value):
        raise ValueError(f"{field} is not a safe path component: {value!r}")
    return value


def _crop_id(section_id: str, scale_index: int, box: CropBox) -> str:
    return (
        f"{section_id}__scale_{scale_index:02d}__"
        f"x_{box.x0:05d}__y_{box.y0:05d}__s_{box.width:05d}"
    )


def _stable_rank(seed: int, crop_id: str) -> tuple[bytes, str]:
    return hashlib.sha256(f"{seed}|{crop_id}".encode("utf-8")).digest(), crop_id


def _select_balanced(
    values: Sequence[PreparedCrop],
    *,
    maximum: int,
    strata_order: Sequence[str],
    random_seed: int,
) -> set[str]:
    eligible = [item for item in values if item.record.eligible]
    if len(eligible) <= maximum:
        return {item.record.crop_id for item in eligible}
    buckets: dict[str, list[PreparedCrop]] = {name: [] for name in strata_order}
    for item in eligible:
        buckets.setdefault(item.record.distance_stratum, []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: _stable_rank(random_seed, item.record.crop_id))
    selected: set[str] = set()
    cursor = {name: 0 for name in buckets}
    active = [name for name in strata_order if buckets.get(name)]
    active.extend(sorted(set(buckets) - set(active)))
    while active and len(selected) < maximum:
        next_active: list[str] = []
        for name in active:
            index = cursor[name]
            if index < len(buckets[name]) and len(selected) < maximum:
                selected.add(buckets[name][index].record.crop_id)
                cursor[name] += 1
            if cursor[name] < len(buckets[name]):
                next_active.append(name)
        active = next_active
    return selected


def prepare_section_crops(
    sample: UruDendro4Sample,
    *,
    split: str,
    source_image: np.ndarray,
    output_root: str | Path,
    crop_config: Mapping[str, Any],
) -> SectionCropPlan:
    """Plan deterministic overlapping multi-scale crops for one source section."""

    split = _safe_component(str(split), field="split")
    if source_image.shape[:2] != (sample.height, sample.width):
        raise ValueError(
            f"source image shape {source_image.shape[:2]} disagrees with indexed "
            f"shape {(sample.height, sample.width)}"
        )
    if crop_config.get("schema_version") not in {None, "racpith.crops.v1"}:
        raise ValueError(f"unsupported crop configuration schema {crop_config.get('schema_version')!r}")
    fractions = [float(value) for value in crop_config["crop_size_fractions_of_min_side"]]
    stride_fraction = float(crop_config["stride_fraction_of_crop"])
    minimum_foreground = float(crop_config["minimum_foreground_fraction"])
    minimum_parent_rings = int(crop_config["minimum_visible_parent_rings"])
    minimum_fragment = float(crop_config["minimum_fragment_length_px"])
    maximum_per_scale = int(crop_config["maximum_crops_per_section_per_size"])
    random_seed = int(crop_config.get("random_seed", 20260906))
    maximum_spacing = float(crop_config.get("maximum_vertex_spacing_px", 4.0))
    if not 0.0 <= minimum_foreground <= 1.0:
        raise ValueError("minimum_foreground_fraction must lie in [0,1]")
    if minimum_parent_rings < 1 or minimum_fragment <= 0.0 or maximum_per_scale < 1:
        raise ValueError("ring/fragment/crop eligibility thresholds must be positive")
    strata = _validate_distance_strata(crop_config["distance_strata"])
    image_format = str(crop_config.get("image_format", "png")).lower().lstrip(".")
    if image_format not in {"png", "jpg", "jpeg"}:
        raise ValueError(f"unsupported crop image format: {image_format!r}")

    foreground = foreground_mask(
        source_image, threshold=int(crop_config.get("foreground_threshold", 3))
    )
    root = Path(output_root).expanduser().resolve()
    inferred_dataset_root = Path(
        os.path.commonpath(
            [str(sample.image_path.resolve().parent), str(sample.annotation_path.resolve().parent)]
        )
    )
    if root == inferred_dataset_root or root.is_relative_to(inferred_dataset_root):
        raise ValueError(
            "output_root must be outside the inferred source dataset root; source data are read-only"
        )
    section_component = _safe_component(sample.section_id, field="section_id")
    prepared: list[PreparedCrop] = []
    boxes = overlapping_crop_boxes(
        sample.image_size_px,
        fractions,
        stride_fraction_of_crop=stride_fraction,
    )
    for scale_index, scale_fraction, box in boxes:
        crop_id = _crop_id(sample.section_id, scale_index, box)
        annotation = build_crop_annotation(
            sample,
            box,
            crop_id=crop_id,
            split=split,
            minimum_fragment_length_px=minimum_fragment,
            maximum_vertex_spacing_px=maximum_spacing,
        )
        foreground_fraction = float(foreground[box.y0 : box.y1, box.x0 : box.x1].mean())
        visible_parent_rings = int(annotation["visible_parent_rings"])
        distance_value = _signed_distance_to_crop(sample.pith_source_px, box)
        distance_stratum = _distance_stratum(distance_value, strata)
        pith_crop_px, pith_inside_crop = crop_pith_coordinates(
            sample.pith_source_px, box
        )
        reasons: list[str] = []
        if foreground_fraction < minimum_foreground:
            reasons.append("LOW_FOREGROUND_FRACTION")
        if visible_parent_rings < minimum_parent_rings:
            reasons.append("TOO_FEW_VISIBLE_PARENT_RINGS")
        if distance_stratum is None:
            reasons.append("DISTANCE_OUTSIDE_CONFIGURED_STRATA")
            distance_stratum = "unassigned"
        output_base = root / split / section_component
        record = CropRecord(
            crop_id=crop_id,
            tree_id=sample.tree_id,
            section_id=sample.section_id,
            split=split,
            box=box,
            scale_index=scale_index,
            scale_fraction=scale_fraction,
            eligible=not reasons,
            selected=False,
            rejection_reasons=tuple(reasons),
            foreground_fraction=foreground_fraction,
            visible_parent_rings=visible_parent_rings,
            distance_value=distance_value,
            distance_stratum=distance_stratum,
            pith_source_px=sample.pith_source_px,
            pith_crop_px=pith_crop_px,
            pith_inside_crop=pith_inside_crop,
            source_image_path=sample.image_path.resolve(),
            source_annotation_path=sample.annotation_path.resolve(),
            crop_image_path=(output_base / "images" / f"{crop_id}.{image_format}").resolve(),
            crop_annotation_path=(output_base / "annotations" / f"{crop_id}.json").resolve(),
        )
        prepared.append(PreparedCrop(record=record, annotation=annotation))

    by_scale: dict[int, list[PreparedCrop]] = {}
    for item in prepared:
        by_scale.setdefault(item.record.scale_index, []).append(item)
    selected_ids: set[str] = set()
    strata_order = [name for name, _, _ in strata]
    for scale_index, values in sorted(by_scale.items()):
        selected_ids.update(
            _select_balanced(
                values,
                maximum=maximum_per_scale,
                strata_order=strata_order,
                random_seed=random_seed + scale_index,
            )
        )
    final = tuple(
        PreparedCrop(
            record=replace(
                item.record, selected=item.record.crop_id in selected_ids and item.record.eligible
            ),
            annotation=item.annotation,
        )
        for item in sorted(prepared, key=lambda value: value.record.crop_id)
    )
    return SectionCropPlan(
        section_id=sample.section_id,
        tree_id=sample.tree_id,
        split=split,
        candidates=final,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def materialize_prepared_crop(
    prepared: PreparedCrop,
    source_image: np.ndarray,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one selected derived crop, never modifying either source file."""

    record = prepared.record
    if not record.eligible or not record.selected:
        raise ValueError(f"crop {record.crop_id} is not an eligible selected crop")
    destinations = (record.crop_image_path, record.crop_annotation_path)
    if record.source_image_path in destinations or record.source_annotation_path in destinations:
        raise ValueError("derived crop destination aliases a source dataset file")
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite derived crop artifacts: {existing}")
    crop_image = prepared.crop_image(source_image)
    extension = record.crop_image_path.suffix.lower()
    success, encoded = cv2.imencode(extension, crop_image)
    if not success:
        raise ValueError(f"OpenCV could not encode crop as {extension}")
    annotation_bytes = (
        json.dumps(
            dict(prepared.annotation),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(record.crop_image_path, encoded.tobytes())
    try:
        _atomic_write_bytes(record.crop_annotation_path, annotation_bytes)
    except BaseException:
        if not record.crop_annotation_path.exists() and not existing:
            try:
                record.crop_image_path.unlink()
            except FileNotFoundError:
                pass
        raise
    return record.to_manifest_row()


def crop_manifest_rows(
    plans: Iterable[SectionCropPlan], *, selected_only: bool = True
) -> Iterable[dict[str, Any]]:
    seen_crop_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for plan in plans:
        for row in plan.manifest_rows(selected_only=selected_only):
            crop_id = str(row["crop_id"])
            paths = {
                Path(str(row["crop_image_path"])).resolve(),
                Path(str(row["crop_annotation_path"])).resolve(),
            }
            if crop_id in seen_crop_ids:
                raise ValueError(f"duplicate crop_id across plans: {crop_id}")
            if seen_paths & paths:
                raise ValueError(f"derived artifact path collision for crop {crop_id}")
            seen_crop_ids.add(crop_id)
            seen_paths.update(paths)
            yield row
