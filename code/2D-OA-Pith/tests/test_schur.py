import torch

from oapith.inference.posterior import marginalize_information


def test_schur_covariance_matches_full_inverse_block() -> None:
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    hessian = matrix.T @ matrix + torch.eye(8, dtype=torch.float64)
    diagnostics = marginalize_information(hessian, 2)
    full_covariance = torch.linalg.inv(hessian)[:2, :2]
    assert diagnostics.rank == 2
    assert diagnostics.covariance is not None
    torch.testing.assert_close(diagnostics.covariance, full_covariance, atol=1e-9, rtol=1e-9)


def test_rank_deficiency_is_reported_not_ridged_away() -> None:
    hessian = torch.diag(torch.tensor([2.0, 0.0, 1.0], dtype=torch.float64))
    diagnostics = marginalize_information(hessian, 2)
    assert diagnostics.rank == 1
    assert diagnostics.covariance is None
    assert diagnostics.condition_number == float("inf")


def test_target_coupling_to_nuisance_null_space_is_invalid() -> None:
    hessian = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    diagnostics = marginalize_information(hessian, 1)
    assert not diagnostics.valid
    assert diagnostics.rank == 0
    assert diagnostics.invalid_reason == "target couples to nuisance null space"


def test_nonfinite_hessian_is_invalid_not_ridged() -> None:
    hessian = torch.eye(3, dtype=torch.float64)
    hessian[0, 0] = torch.nan
    diagnostics = marginalize_information(hessian, 2)
    assert not diagnostics.valid
    assert diagnostics.rank == 0
    assert diagnostics.invalid_reason == "non-finite Hessian"


def test_indefinite_nuisance_hessian_is_invalid() -> None:
    hessian = torch.diag(torch.tensor([2.0, 1.0, -0.5], dtype=torch.float64))
    diagnostics = marginalize_information(hessian, 2)
    assert not diagnostics.valid
    assert diagnostics.invalid_reason == "indefinite nuisance Hessian"
