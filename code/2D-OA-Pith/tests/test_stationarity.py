import torch

from oapith.inference.posterior import _scaled_stationarity


def test_stationarity_uses_effective_sample_scale_not_objective_offset() -> None:
    parameter = torch.tensor([2.0], dtype=torch.float64)
    gradient = (torch.tensor([0.02], dtype=torch.float64),)
    value = _scaled_stationarity(
        [parameter], gradient, torch.tensor(100.0, dtype=torch.float64)
    )
    # max(|g| * max(1, |theta|)) / N_eff.  No objective value appears, so adding
    # an arbitrary likelihood constant cannot change validity.
    torch.testing.assert_close(value, torch.tensor(4e-4, dtype=torch.float64))


def test_stationarity_rejects_nonfinite_gradient() -> None:
    parameter = torch.tensor([1.0], dtype=torch.float64)
    value = _scaled_stationarity(
        [parameter],
        (torch.tensor([float("nan")], dtype=torch.float64),),
        torch.tensor(1.0, dtype=torch.float64),
    )
    assert not bool(torch.isfinite(value))
