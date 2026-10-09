import math

import torch
import torch.nn.functional as functional

from oapith.data.transforms import RandomSimilarity
from oapith.geometry.coordinates import (
    apply_homography,
    normalized_sampling_grid,
    similarity_matrix,
)
from oapith.types import CoordinateFrame


def test_out_of_frame_round_trip_is_not_clipped() -> None:
    frame = CoordinateFrame(640, 480)
    points_px = torch.tensor([[20.0, 30.0], [2400.0, -900.0]], dtype=torch.float64)
    normalized = frame.pixel_to_normalized(points_px)
    assert normalized[1].abs().max() > 1.0
    torch.testing.assert_close(frame.normalized_to_pixel(normalized), points_px)


def test_similarity_applies_to_every_geometry_point() -> None:
    points = torch.tensor([[0.2, -0.3], [4.0, 2.0]], dtype=torch.float64)
    transform = similarity_matrix(math.pi / 2, 1.2, (0.1, -0.2), dtype=torch.float64)
    inverse = torch.linalg.inv(transform)
    restored = apply_homography(apply_homography(points, transform), inverse)
    torch.testing.assert_close(restored, points, atol=1e-12, rtol=1e-12)


def test_covariance_and_curvature_scaling() -> None:
    frame = CoordinateFrame(1000, 500)
    covariance = torch.tensor([[4.0, 1.0], [1.0, 9.0]])
    normalized = frame.covariance_pixel_to_normalized(covariance)
    torch.testing.assert_close(frame.covariance_normalized_to_pixel(normalized), covariance)
    curvature = torch.tensor(0.02)
    torch.testing.assert_close(
        frame.curvature_normalized_to_pixel(frame.curvature_pixel_to_normalized(curvature)),
        curvature,
    )


def test_composed_native_grid_samples_original_pixels_once() -> None:
    # A 4x magnification from a 64-pixel source to a 16-pixel output should read
    # the central 16 source pixels one-for-one.  This fails if the source is first
    # reduced to a 16-pixel full-image square and then magnified.
    source = torch.arange(64 * 64, dtype=torch.float64).reshape(1, 1, 64, 64)
    transform = similarity_matrix(0.0, 4.0, dtype=torch.float64)
    grid, valid = normalized_sampling_grid(
        CoordinateFrame(64, 64),
        16,
        forward_matrix=transform,
        dtype=torch.float64,
    )
    sampled = functional.grid_sample(
        source, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch.testing.assert_close(sampled, source[..., 24:40, 24:40], atol=1e-10, rtol=0)
    assert bool(valid.all())


def test_rotated_native_crop_corners_stay_in_rectangular_content() -> None:
    augmentation = RandomSimilarity(max_rotation_radians=math.pi)
    generator = torch.Generator().manual_seed(2026)
    content_half_extent = (1.0, 0.6)
    output_corners = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    for _ in range(32):
        transform = augmentation.sample_native_view(
            4.0,
            jitter=(1.0, 1.0),
            content_half_extent=content_half_extent,
            generator=generator,
            dtype=torch.float64,
        )
        source_corners = apply_homography(output_corners, torch.linalg.inv(transform))
        assert bool(source_corners[:, 0].abs().max() <= content_half_extent[0] + 1e-12)
        assert bool(source_corners[:, 1].abs().max() <= content_half_extent[1] + 1e-12)
