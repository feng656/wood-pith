import torch

from oapith.calibration.conformal import (
    ConformalCalibrator,
    ConformalComponent,
    MixturePrediction,
    joint_state_decision,
    joint_state_probabilities,
    prediction_from_posteriors,
)
from oapith.inference.posterior import InformationDiagnostics, ModePosterior
from oapith.inference.refiner import RefinementMode, RefinementResult
from oapith.types import PithState


def test_grouped_conformal_uses_group_maximum() -> None:
    prediction = MixturePrediction(
        [ConformalComponent("near", 1.0, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])]
    )
    predictions = [prediction] * 4
    truth = [
        torch.tensor([0.1, 0.0]),
        torch.tensor([2.0, 0.0]),
        torch.tensor([0.2, 0.0]),
        torch.tensor([0.3, 0.0]),
    ]
    groups = ["disc-a", "disc-a", "disc-b", "disc-c"]
    calibrator = ConformalCalibrator(alpha=0.5, reduction="max").fit(
        predictions, truth, groups
    )
    assert calibrator.number_groups == 3
    assert calibrator.threshold is not None


def test_small_calibration_set_uses_infinity_atom() -> None:
    prediction = MixturePrediction(
        [ConformalComponent("near", 1.0, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])]
    )
    calibrator = ConformalCalibrator(alpha=0.01).fit(
        [prediction, prediction],
        [torch.zeros(2), torch.ones(2)],
        ["a", "b"],
    )
    assert calibrator.threshold == float("inf")


def test_joint_state_mass_includes_unresolved_null_and_full_mixture() -> None:
    prediction = MixturePrediction(
        [
            ConformalComponent("near", 0.25, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
            ConformalComponent(
                "far_unbounded",
                0.35,
                [0.0, 0.1],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ),
            ConformalComponent(
                "infinity", 0.10, [0.0, 0.0], [[1.0, 0.0], [0.0, 0.01]]
            ),
        ],
        null_probability=0.30,
    )
    probabilities = joint_state_probabilities(prediction)
    torch.testing.assert_close(
        torch.tensor(probabilities), torch.tensor([0.25, 0.35, 0.10, 0.30])
    )


def test_joint_null_decision_is_shared_with_conformal_contains() -> None:
    prediction = MixturePrediction(
        [
            ConformalComponent("near", 0.30, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
            ConformalComponent("far", 0.15, [0.0, 0.2], [[1.0, 0.0], [0.0, 1.0]]),
            ConformalComponent(
                "infinity", 0.10, [0.0, 0.0], [[1.0, 0.0], [0.0, 0.01]]
            ),
        ],
        null_probability=0.45,
    )
    probabilities = joint_state_probabilities(prediction)
    assert joint_state_decision(probabilities) is PithState.NULL
    assert bool(prediction.contains(torch.tensor([100.0, 100.0]), threshold=-1e6))


def test_joint_null_wins_exact_tie_conservatively() -> None:
    assert joint_state_decision([0.5, 0.0, 0.0, 0.5]) is PithState.NULL


def _information(*, valid: bool) -> InformationDiagnostics:
    return InformationDiagnostics(
        information=torch.eye(2, dtype=torch.float64),
        covariance=torch.eye(2, dtype=torch.float64) if valid else None,
        rank=2 if valid else 0,
        dimension=2,
        minimum_eigenvalue=1.0,
        maximum_eigenvalue=1.0,
        condition_number=1.0 if valid else float("inf"),
        pseudo_logdet=0.0 if valid else -float("inf"),
        nuisance_rank=0,
        nuisance_dimension=0,
        positive_semidefinite=valid,
        valid=valid,
        invalid_reason=None if valid else "test-invalid",
    )


def _posterior(weight: float, valid: bool, x: float) -> ModePosterior:
    result = RefinementResult(
        mode=RefinementMode.NEAR,
        state=PithState.NEAR,
        center=torch.tensor([x, 0.0], dtype=torch.float64),
        direction=torch.tensor([1.0, 0.0], dtype=torch.float64),
        inverse_distance=1.0 / max(x, 1.0),
        loss=0.0,
        reliabilities={},
        model_state={},
        evidence_score=0.0,
    )
    information = _information(valid=valid)
    return ModePosterior(result, weight, information, information)


def test_invalid_laplace_mass_moves_to_null() -> None:
    prediction = prediction_from_posteriors(
        [_posterior(0.6, True, 0.2), _posterior(0.4, False, 0.3)],
        null_probability=0.1,
    )
    # q_null + (1-q_null)*unresolved = 0.1 + 0.9*0.4
    assert abs(prediction.null_probability - 0.46) < 1e-12
    probabilities = joint_state_probabilities(prediction)
    assert abs(probabilities[int(PithState.NEAR)] - 0.54) < 1e-12
    assert abs(probabilities[int(PithState.NULL)] - 0.46) < 1e-12
