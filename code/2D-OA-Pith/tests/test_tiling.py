import torch
from torch import nn

from oapith.inference.tiling import (
    tile_distance_truncation_in_global_chart,
    tiled_dense_prediction,
)
from oapith.types import CoordinateFrame


class _ConstantDenseModel(nn.Module):
    def forward(
        self, image: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        batch, _, height, width = image.shape
        scalar = image[:, :1].mean(dim=(-2, -1), keepdim=True)
        mean = scalar.expand(batch, 1, height, width)
        log_variance = torch.zeros_like(mean)
        zeros = torch.zeros_like(mean)
        orientation = torch.cat([torch.ones_like(mean), zeros], 1)
        embedding = orientation.clone()
        return {
            "ring_logits": zeros,
            "distance": mean,
            "log_variance": log_variance,
            "orientation": orientation,
            "embedding": embedding,
            "defect_logits": zeros,
        }


def test_tile_blending_uses_second_moments_for_variance() -> None:
    image = torch.zeros(1, 3, 24, 32)
    image[..., :, 16:] = 2.0
    result = tiled_dense_prediction(
        _ConstantDenseModel(), image, tile_size=16, overlap=8, batch_size=2
    )
    variance = torch.exp(result["log_variance"])
    # Every component has unit variance. Disagreement between overlapping tile
    # means must add, never reduce, predictive variance.
    assert bool((variance >= 1.0 - 1e-6).all())
    assert float(variance[..., 8:16, 12:20].max()) > 1.0


def test_tile_uncertainty_converts_to_global_normalized_units() -> None:
    truncation = 0.04
    torch.testing.assert_close(
        torch.tensor(
            tile_distance_truncation_in_global_chart(
                truncation, 768, CoordinateFrame(768, 512)
            )
        ),
        torch.tensor(truncation),
    )
    torch.testing.assert_close(
        torch.tensor(
            tile_distance_truncation_in_global_chart(
                truncation, 768, CoordinateFrame(1536, 512)
            )
        ),
        torch.tensor(truncation / 2),
    )
