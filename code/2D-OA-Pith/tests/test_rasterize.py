import torch

from oapith.data.rasterize import NormalizedCurve, rasterize_curves


def test_soft_curve_retains_subpixel_uncertainty() -> None:
    curve = NormalizedCurve(
        ring_id=3,
        points=torch.tensor([[-0.8, 0.1], [0.0, 0.1], [0.8, 0.1]]),
        sigma=torch.tensor(0.01),
    )
    target = rasterize_curves([curve], 128)
    assert target["ring_probability"].max() > 0.5
    assert target["ring_index"].eq(3).any()
    assert target["annotation_sigma_distance"].shape == (1, 128, 128)
    orientation = target["orientation"][:, 70:72, 60:68]
    assert orientation[0].mean() > 0.8  # horizontal tangent -> cos(2 theta)=1
