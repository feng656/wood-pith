from __future__ import annotations

import torch
import torch.nn.functional as functional

from oapith.types import CoordinateFrame


def tile_distance_truncation_in_global_chart(
    raster_truncation: float,
    tile_size: int,
    frame: CoordinateFrame,
) -> float:
    """Convert a tile-chart truncated-distance unit to global normalized length."""
    if raster_truncation <= 0 or tile_size < 1:
        raise ValueError("truncation and tile_size must be positive")
    # One tile-chart unit spans tile_size/2 source pixels.  One global-chart
    # unit spans frame.scale_px=max(width,height)/2 pixels.
    return float(raster_truncation) * tile_size / (2.0 * frame.scale_px)


def _starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= overlap or tile < 16:
        raise ValueError("tile must exceed overlap and be at least 16")
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, tile - overlap))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.inference_mode()
def tiled_dense_prediction(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    tile_size: int = 1024,
    overlap: int = 192,
    batch_size: int = 2,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Blend native-resolution dense predictions without shrinking fine rings."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape [1,C,H,W]")
    _, _, height, width = image.shape
    ys, xs = _starts(height, tile_size, overlap), _starts(width, tile_size, overlap)
    windows = [(y, x) for y in ys for x in xs]
    one_dimensional = torch.hann_window(tile_size, periodic=False, device=image.device).clamp_min(0.05)
    blending = (one_dimensional[:, None] * one_dimensional[None, :])[None, None]
    accumulators: dict[str, torch.Tensor] = {}
    weight = image.new_zeros(1, 1, height, width)
    dense_keys = {
        "ring_logits",
        "distance",
        "log_variance",
        "orientation",
        "embedding",
        "defect_logits",
    }
    for group_start in range(0, len(windows), batch_size):
        group = windows[group_start : group_start + batch_size]
        tiles, masks, shapes, offsets = [], [], [], []
        for y, x in group:
            tile = image[..., y : min(y + tile_size, height), x : min(x + tile_size, width)]
            shape = tile.shape[-2:]
            pad_x = tile_size - shape[1]
            pad_y = tile_size - shape[0]
            left, top = pad_x // 2, pad_y // 2
            pad = (left, pad_x - left, top, pad_y - top)
            # Inputs are already standardized, so constant zero is the same
            # padding convention used by training's normalized square.
            tiles.append(functional.pad(tile, pad, mode="constant", value=0.0))
            if valid_mask is None:
                mask = tile.new_ones(1, 1, *shape)
            else:
                mask = valid_mask[..., y : y + shape[0], x : x + shape[1]]
            masks.append(functional.pad(mask, pad))
            shapes.append(shape)
            offsets.append((top, left))
        batch = torch.cat(tiles, 0)
        mask_batch = torch.cat(masks, 0)
        output = model(batch, mask_batch)
        for local, ((y, x), shape, (top, left)) in enumerate(
            zip(group, shapes, offsets)
        ):
            rows = slice(top, top + shape[0])
            columns = slice(left, left + shape[1])
            local_weight = blending[..., rows, columns] * masks[local][..., rows, columns]
            weight[..., y : y + shape[0], x : x + shape[1]] += local_weight
            for key in dense_keys:
                value = output[key][local : local + 1, ..., rows, columns]
                if key == "log_variance":
                    # Blend predictive second moments, not log variances.  This
                    # propagates disagreement between overlapping tile means into
                    # the final localization variance instead of averaging it away.
                    local_mean = output["distance"][local : local + 1, ..., rows, columns]
                    value = torch.exp(value) + local_mean.square()
                if key not in accumulators:
                    accumulators[key] = image.new_zeros(1, value.shape[1], height, width)
                accumulators[key][..., y : y + shape[0], x : x + shape[1]] += value * local_weight
    for key in accumulators:
        accumulators[key] = accumulators[key] / weight.clamp_min(1e-6)
    predictive_variance = (
        accumulators["log_variance"] - accumulators["distance"].square()
    ).clamp_min(1e-8)
    accumulators["log_variance"] = torch.log(predictive_variance).clamp(-8.0, 6.0)
    accumulators["orientation"] = torch.nn.functional.normalize(
        accumulators["orientation"], dim=1, eps=1e-8
    )
    accumulators["embedding"] = torch.nn.functional.normalize(
        accumulators["embedding"], dim=1, eps=1e-8
    )
    accumulators["valid_mask"] = weight > 0
    return accumulators
