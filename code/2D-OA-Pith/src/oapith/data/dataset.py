from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import Dataset

from oapith.data.manifest import SampleRecord, load_manifest
from oapith.data.rasterize import NormalizedCurve, rasterize_curves
from oapith.data.transforms import RandomSimilarity
from oapith.geometry.coordinates import (
    apply_homography,
    normalized_sampling_grid,
    transform_covariance,
    transform_tangents,
)
from oapith.geometry.curves import estimate_tangents
from oapith.types import CoordinateFrame
from oapith.types import PithState


def _read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _read_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class PithDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path | list[SampleRecord],
        *,
        root: str | Path = ".",
        split: str | None = None,
        image_size: int = 512,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.25, 0.25, 0.25),
        augment: RandomSimilarity | None = None,
        raster_truncation: float = 0.04,
        state_switch_radius: float = 2.0,
        native_tile_probability: float = 0.5,
        native_scale_jitter: tuple[float, float] = (0.8, 1.25),
    ) -> None:
        self.records = load_manifest(manifest, split) if not isinstance(manifest, list) else manifest
        self.root = Path(root)
        self.image_size = int(image_size)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        if (self.std <= 0).any():
            raise ValueError("RGB standard deviations must be positive")
        self.augment = augment
        self.raster_truncation = raster_truncation
        if state_switch_radius <= 0:
            raise ValueError("state_switch_radius must be positive")
        self.state_switch_radius = float(state_switch_radius)
        if not 0.0 <= native_tile_probability <= 1.0:
            raise ValueError("native_tile_probability must lie in [0,1]")
        if native_scale_jitter[0] <= 0 or native_scale_jitter[1] < native_scale_jitter[0]:
            raise ValueError("invalid native_scale_jitter")
        self.native_tile_probability = float(native_tile_probability)
        self.native_scale_jitter = tuple(float(value) for value in native_scale_jitter)

    def __len__(self) -> int:
        return len(self.records)

    def _normalize_curves(
        self, record: SampleRecord, frame: CoordinateFrame
    ) -> list[NormalizedCurve]:
        result: list[NormalizedCurve] = []
        for index, curve in enumerate(record.curves):
            points = frame.pixel_to_normalized(torch.tensor(curve.points_px, dtype=torch.float32))
            raw_sigma = torch.as_tensor(curve.sigma_px, dtype=torch.float32)
            sigma = raw_sigma / frame.scale_px
            result.append(
                NormalizedCurve(
                    ring_id=curve.ring_id,
                    points=points,
                    sigma=sigma,
                    visibility=curve.visibility,
                    arc_id=curve.arc_id or f"{record.sample_id}:{index}",
                )
            )
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = _read_rgb(self.root / record.image)
        # Standardize before geometric sampling so that out-of-image padding is
        # exactly zero in both training and tiled inference.  Standardizing the
        # padded square afterwards would turn padding into -mean/std and create a
        # train/inference boundary cue.
        image = (image - self.mean) / self.std
        height, width = image.shape[-2:]
        spacing = record.mm_per_pixel or [None, None]
        frame = CoordinateFrame(width, height, spacing[0], spacing[1])
        curves = self._normalize_curves(record, frame)
        pith = (
            frame.pixel_to_normalized(torch.tensor(record.pith_px, dtype=torch.float32))
            if record.pith_px is not None
            else torch.zeros(2)
        )
        pith_covariance = (
            frame.covariance_pixel_to_normalized(
                torch.tensor(record.pith_cov_px, dtype=torch.float32)
            )
            if record.pith_cov_px is not None
            else torch.eye(2) * (1.0 / frame.scale_px) ** 2
        )
        augmentation = torch.eye(3)
        state_target = record.pith_state if record.pith_state is not None else -1
        if self.augment is not None:
            if torch.rand(()).item() < self.native_tile_probability:
                augmentation = self.augment.sample_native_view(
                    max(width, height) / self.image_size,
                    jitter=self.native_scale_jitter,
                    content_half_extent=(
                        width / max(width, height),
                        height / max(width, height),
                    ),
                    dtype=image.dtype,
                )
            else:
                augmentation = self.augment.sample(dtype=image.dtype)
            scale = torch.sqrt(torch.det(augmentation[:2, :2]).abs())
            for curve in curves:
                tangent = estimate_tangents(curve.points)
                curve.points = apply_homography(curve.points, augmentation)
                # Calculated to make the transformation of vector data explicit.
                _ = transform_tangents(tangent, augmentation)
                curve.sigma = curve.sigma * scale
            if record.pith_state == int(PithState.INFINITY):
                # Infinity labels are direction proxies: translation must not turn
                # them into finite points or rotate them incorrectly.
                pith = pith @ augmentation[:2, :2].T
            else:
                pith = apply_homography(pith.unsqueeze(0), augmentation).squeeze(0)
            pith_covariance = transform_covariance(pith_covariance, augmentation)
            # Similarity changes finite distance in the normalized chart, so update
            # the finite chart label.  Infinity remains infinity under a similarity;
            # a NULL crop cannot gain unseen pixels through this warp.  This keeps
            # state supervision alive instead of silently training it only by C4.
            if record.pith_state in (int(PithState.INFINITY), int(PithState.NULL)):
                state_target = int(record.pith_state)
            elif record.pith_px is not None:
                visible_ring_evidence = any(
                    bool(((curve.points.abs() <= 1.0).all(dim=-1)).any())
                    for curve in curves
                )
                state_target = (
                    int(
                        PithState.NEAR
                        if torch.linalg.vector_norm(pith) <= self.state_switch_radius
                        else PithState.FAR
                    )
                    if visible_ring_evidence
                    else -1
                )
            else:
                state_target = -1

        # Compose augmentation^-1 with square-chart -> rectangular-source
        # coordinates and sample the original tensor once.  In particular, a
        # native-scale crop must not magnify an already downsampled full-image
        # square, because that cannot recover thin-ring frequencies.
        grid, valid = normalized_sampling_grid(
            frame,
            self.image_size,
            forward_matrix=augmentation,
            dtype=image.dtype,
        )
        square = functional.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
        if record.unknown_mask is not None:
            unknown = _read_mask(self.root / record.unknown_mask)
            if unknown.shape[-2:] != (height, width):
                raise ValueError(
                    f"unknown mask shape {unknown.shape[-2:]} does not match image {(height, width)}"
                )
            unknown_square = functional.grid_sample(
                unknown.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            valid = valid * (1.0 - unknown_square).clamp(0, 1)

        dense = (
            rasterize_curves(
                curves,
                self.image_size,
                truncation=self.raster_truncation,
            )
            if curves
            else self._empty_dense()
        )
        return {
            "image": square,
            "valid_mask": valid.squeeze(0),
            "dense": dense,
            "pith": pith,
            "pith_covariance": pith_covariance,
            "pith_valid": torch.tensor(record.pith_px is not None),
            "rings_annotated": torch.tensor(record.rings_annotated),
            "state_target": torch.tensor(state_target),
            "curves": curves,
            "meta": {
                "sample_id": record.sample_id,
                "group_id": record.group_id,
                "image_path": str(self.root / record.image),
                "frame": frame.as_dict(),
                "augmentation": augmentation.tolist(),
                "pith_px": record.pith_px,
            },
        }

    def _empty_dense(self) -> dict[str, torch.Tensor]:
        s = self.image_size
        return {
            "ring_probability": torch.zeros(1, s, s),
            "distance": torch.ones(1, s, s),
            "orientation": torch.zeros(2, s, s),
            "orientation_valid": torch.zeros(1, s, s),
            "ring_index": torch.full((s, s), -1, dtype=torch.long),
            "annotation_sigma": torch.ones(1, s, s) * (2.0 / s),
            "annotation_sigma_distance": torch.ones(1, s, s) * (2.0 / s) / self.raster_truncation,
            "label_weight": torch.zeros(1, s, s),
        }


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    dense_keys = samples[0]["dense"].keys()
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "valid_mask": torch.stack([sample["valid_mask"] for sample in samples]),
        "dense": {
            key: torch.stack([sample["dense"][key] for sample in samples]) for key in dense_keys
        },
        "pith": torch.stack([sample["pith"] for sample in samples]),
        "pith_covariance": torch.stack([sample["pith_covariance"] for sample in samples]),
        "pith_valid": torch.stack([sample["pith_valid"] for sample in samples]),
        "rings_annotated": torch.stack([sample["rings_annotated"] for sample in samples]),
        "state_target": torch.stack([sample["state_target"] for sample in samples]),
        "curves": [sample["curves"] for sample in samples],
        "meta": [sample["meta"] for sample in samples],
    }
