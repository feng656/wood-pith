import torch

from oapith.inference.refiner import FarLayerModel


def test_rationalized_far_residual_is_continuous_at_infinity() -> None:
    points = torch.tensor([[0.2, 0.4], [-0.7, 0.1], [0.9, -0.3]], dtype=torch.float64)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
    at_infinity, gradient = FarLayerModel._base_factor_and_gradient(
        points, direction, torch.tensor(0.0, dtype=torch.float64)
    )
    torch.testing.assert_close(at_infinity, -points[:, 0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        gradient,
        torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    near_infinity, _ = FarLayerModel._base_factor_and_gradient(
        points, direction, torch.tensor(1e-8, dtype=torch.float64)
    )
    torch.testing.assert_close(near_infinity, at_infinity, atol=2e-8, rtol=0)
