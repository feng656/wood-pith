from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .provenance import atomic_write_json, read_json_object


def _write_deterministic_npz(
    handle: Any,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Write an NPZ whose bytes do not depend on wall-clock ZIP metadata."""

    with zipfile.ZipFile(
        handle,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, value in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asanyarray(value),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


class GeometryState(StrEnum):
    POINT = "POINT"
    RANGE = "RANGE"
    RAY = "RAY"
    AXIS = "AXIS"
    MULTIMODAL = "MULTIMODAL"
    REJECT = "REJECT"


@dataclass(frozen=True)
class EvidenceBundle:
    """Candidate-independent quadrature evidence for one crop.

    ``base_weight`` is the frozen parent-ring evidence measure.  An active-mask
    deletion drops entries without renormalising their siblings.
    """

    crop_id: str
    tree_id: str
    section_id: str
    points_norm: np.ndarray
    points_crop_px: np.ndarray
    tangents: np.ndarray
    sigma_x_norm: np.ndarray
    sigma_alg_norm2: np.ndarray
    base_weight: np.ndarray
    ring_index: np.ndarray
    arc_index: np.ndarray
    source_s_px: np.ndarray
    arc_fraction: np.ndarray
    quality: np.ndarray
    ring_ids: tuple[str, ...]
    arc_ids: tuple[str, ...]
    crop_origin_source_px: tuple[float, float]
    crop_size_px: tuple[int, int]
    normalization_scale_px: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.points_norm.shape[0])

    def validate(self) -> None:
        n = self.n_nodes
        vector_fields = {
            "points_crop_px": (n, 2),
            "tangents": (n, 2),
        }
        if self.points_norm.shape != (n, 2):
            raise ValueError("points_norm must have shape (N, 2)")
        for name, shape in vector_fields.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        for name in (
            "sigma_x_norm",
            "sigma_alg_norm2",
            "base_weight",
            "ring_index",
            "arc_index",
            "source_s_px",
            "arc_fraction",
            "quality",
        ):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        numeric = (
            self.points_norm,
            self.points_crop_px,
            self.tangents,
            self.sigma_x_norm,
            self.sigma_alg_norm2,
            self.base_weight,
            self.source_s_px,
            self.arc_fraction,
            self.quality,
        )
        if not all(np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("evidence contains NaN or infinity")
        if np.any(self.sigma_x_norm <= 0) or np.any(self.sigma_alg_norm2 <= 0):
            raise ValueError("all measurement scales must be strictly positive")
        if np.any(self.base_weight <= 0):
            raise ValueError("all evidence weights must be strictly positive")
        if np.any(self.arc_fraction < 0) or np.any(self.arc_fraction > 1):
            raise ValueError("arc_fraction must lie in [0, 1]")
        tangent_norm = np.linalg.norm(self.tangents, axis=1)
        if np.max(np.abs(tangent_norm - 1.0), initial=0.0) > 1e-5:
            raise ValueError("tangents must be unit vectors")
        if np.min(self.ring_index, initial=0) < 0 or np.max(
            self.ring_index, initial=-1
        ) >= len(self.ring_ids):
            raise ValueError("ring_index is outside ring_ids")
        if np.min(self.arc_index, initial=0) < 0 or np.max(
            self.arc_index, initial=-1
        ) >= len(self.arc_ids):
            raise ValueError("arc_index is outside arc_ids")
        if self.normalization_scale_px <= 0:
            raise ValueError("normalization_scale_px must be positive")
        width, height = self.crop_size_px
        if width <= 0 or height <= 0:
            raise ValueError("crop dimensions must be positive")
        if n and (
            np.any(self.points_crop_px[:, 0] < 0.0)
            or np.any(self.points_crop_px[:, 0] >= float(width))
            or np.any(self.points_crop_px[:, 1] < 0.0)
            or np.any(self.points_crop_px[:, 1] >= float(height))
        ):
            raise ValueError(
                "quadrature evidence leaves the half-open crop coordinate domain"
            )
        if len(self.crop_origin_source_px) != 2 or not np.all(
            np.isfinite(np.asarray(self.crop_origin_source_px, dtype=np.float64))
        ):
            raise ValueError("crop_origin_source_px must contain two finite values")
        expected_norm = (
            self.points_crop_px
            - np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
        ) / float(self.normalization_scale_px)
        if not np.allclose(
            self.points_norm,
            expected_norm,
            rtol=0.0,
            atol=2.0e-10,
        ):
            raise ValueError(
                "points_norm and points_crop_px violate the registered crop-coordinate transform"
            )
        ring_mass = np.bincount(
            self.ring_index, weights=self.base_weight, minlength=len(self.ring_ids)
        )
        positive = ring_mass[ring_mass > 0]
        if positive.size == 0:
            raise ValueError("no parent-ring evidence")
        expected = 1.0 / positive.size
        if np.max(np.abs(positive - expected)) > max(1e-8, expected * 1e-5):
            raise ValueError(
                "parent-ring budgets are not equal; sampling or splitting may have changed evidence mass"
            )

    def save(self, npz_path: str | Path, metadata_path: str | Path) -> None:
        self.validate()
        npz_target = Path(npz_path).expanduser().resolve()
        metadata_target = Path(metadata_path).expanduser().resolve()
        if npz_target == metadata_target:
            raise ValueError("evidence NPZ and metadata paths must be different")
        if npz_target.parent != metadata_target.parent:
            raise ValueError("evidence NPZ and metadata must be stored side by side")
        npz_target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{npz_target.name}.", suffix=".npz", dir=npz_target.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                _write_deterministic_npz(
                    handle,
                    {
                        "points_norm": self.points_norm,
                        "points_crop_px": self.points_crop_px,
                        "tangents": self.tangents,
                        "sigma_x_norm": self.sigma_x_norm,
                        "sigma_alg_norm2": self.sigma_alg_norm2,
                        "base_weight": self.base_weight,
                        "ring_index": self.ring_index,
                        "arc_index": self.arc_index,
                        "source_s_px": self.source_s_px,
                        "arc_fraction": self.arc_fraction,
                        "quality": self.quality,
                    },
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, npz_target)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        payload = {
            "schema_version": "racpith.evidence.v1",
            "crop_id": self.crop_id,
            "tree_id": self.tree_id,
            "section_id": self.section_id,
            "ring_ids": list(self.ring_ids),
            "arc_ids": list(self.arc_ids),
            "crop_origin_source_px": list(self.crop_origin_source_px),
            "crop_size_px": list(self.crop_size_px),
            "normalization_scale_px": self.normalization_scale_px,
            "metadata": dict(self.metadata),
            "npz_path": npz_target.name,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{metadata_target.name}.", dir=metadata_target.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, metadata_target)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, metadata_path: str | Path) -> "EvidenceBundle":
        meta_path = Path(metadata_path)
        payload = read_json_object(meta_path)
        if payload.get("schema_version") != "racpith.evidence.v1":
            raise ValueError(f"unsupported evidence schema in {meta_path}")
        npz_name = Path(str(payload["npz_path"]))
        if npz_name.is_absolute() or npz_name.name != str(npz_name):
            raise ValueError("evidence npz_path must be a basename next to its metadata")
        with np.load(meta_path.parent / npz_name, allow_pickle=False) as arrays:
            # Materialise every member before closing the archive.  This keeps the
            # returned bundle independent of an open ZipFile/file descriptor.
            bundle = cls(
                crop_id=payload["crop_id"],
                tree_id=payload["tree_id"],
                section_id=payload["section_id"],
                points_norm=np.array(arrays["points_norm"], copy=True),
                points_crop_px=np.array(arrays["points_crop_px"], copy=True),
                tangents=np.array(arrays["tangents"], copy=True),
                sigma_x_norm=np.array(arrays["sigma_x_norm"], copy=True),
                sigma_alg_norm2=np.array(arrays["sigma_alg_norm2"], copy=True),
                base_weight=np.array(arrays["base_weight"], copy=True),
                ring_index=np.array(arrays["ring_index"], dtype=np.int64, copy=True),
                arc_index=np.array(arrays["arc_index"], dtype=np.int64, copy=True),
                source_s_px=np.array(arrays["source_s_px"], copy=True),
                arc_fraction=np.array(arrays["arc_fraction"], copy=True),
                quality=np.array(arrays["quality"], copy=True),
                ring_ids=tuple(payload["ring_ids"]),
                arc_ids=tuple(payload["arc_ids"]),
                crop_origin_source_px=tuple(payload["crop_origin_source_px"]),
                crop_size_px=tuple(payload["crop_size_px"]),
                normalization_scale_px=float(payload["normalization_scale_px"]),
                metadata=payload.get("metadata", {}),
            )
        bundle.validate()
        return bundle


@dataclass
class LocateResult:
    crop_id: str
    state: GeometryState
    raw_center_norm: list[float] | None
    usable_center_norm: list[float] | None
    direction_norm: list[float] | None
    range_interval_norm: list[float | None] | None
    objective: float | None
    radii_norm: Mapping[str, float]
    condition_ratio: float | None
    weak_direction: list[float] | None
    finite_modes: list[Mapping[str, Any]]
    far_scan: Mapping[str, Any]
    profiles: list[Mapping[str, Any]]
    search_adequate: bool
    model_risk: str
    production_usable: bool
    reason_codes: list[str]
    diagnostics: Mapping[str, Any]
    config_hash: str
    run_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "racpith.locate.v1",
            "crop_id": self.crop_id,
            "state": self.state.value,
            "raw_center_norm": self.raw_center_norm,
            "usable_center_norm": self.usable_center_norm,
            "direction_norm": self.direction_norm,
            "range_interval_norm": self.range_interval_norm,
            "objective": self.objective,
            "radii_norm": dict(self.radii_norm),
            "condition_ratio": self.condition_ratio,
            "weak_direction": self.weak_direction,
            "finite_modes": list(self.finite_modes),
            "far_scan": dict(self.far_scan),
            "profiles": list(self.profiles),
            "search_adequate": self.search_adequate,
            "model_risk": self.model_risk,
            "production_usable": self.production_usable,
            "reason_codes": list(self.reason_codes),
            "diagnostics": dict(self.diagnostics),
            "config_hash": self.config_hash,
            "run_id": self.run_id,
        }

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.as_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LocateResult":
        if payload.get("schema_version") != "racpith.locate.v1":
            raise ValueError("unsupported localization-result schema")
        return cls(
            crop_id=str(payload["crop_id"]),
            state=GeometryState(str(payload["state"])),
            raw_center_norm=payload.get("raw_center_norm"),
            usable_center_norm=payload.get("usable_center_norm"),
            direction_norm=payload.get("direction_norm"),
            range_interval_norm=payload.get("range_interval_norm"),
            objective=payload.get("objective"),
            radii_norm=payload.get("radii_norm", {}),
            condition_ratio=payload.get("condition_ratio"),
            weak_direction=payload.get("weak_direction"),
            finite_modes=payload.get("finite_modes", []),
            far_scan=payload.get("far_scan", {}),
            profiles=payload.get("profiles", []),
            search_adequate=bool(payload.get("search_adequate", False)),
            model_risk=str(payload.get("model_risk", "UNKNOWN")),
            production_usable=bool(payload.get("production_usable", False)),
            reason_codes=list(payload.get("reason_codes", [])),
            diagnostics=payload.get("diagnostics", {}),
            config_hash=str(payload["config_hash"]),
            run_id=str(payload.get("run_id", "UNSPECIFIED")),
        )


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [json_safe(item) for item in value]
    return value
