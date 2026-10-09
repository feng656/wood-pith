from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from oapith.inference.posterior import ModePosterior
from oapith.inference.refiner import RefinementMode
from oapith.types import PithState


def _wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _observable_precision(information: torch.Tensor) -> torch.Tensor | None:
    """PSD projection for a data-supported (possibly rank-deficient) chart."""
    symmetric = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    scale = eigenvalues.abs().max().clamp_min(1.0)
    tolerance = 1e-8 * scale
    if bool((eigenvalues < -tolerance).any()):
        return None
    positive = eigenvalues > tolerance
    if not bool(positive.any()):
        return torch.zeros_like(symmetric)
    vectors = eigenvectors[:, positive]
    return vectors @ torch.diag(eigenvalues[positive]) @ vectors.T


@dataclass
class ConformalComponent:
    kind: str
    weight: float
    mean: list[float]
    covariance: list[list[float]]
    precision: list[list[float]] | None = None


@dataclass
class MixturePrediction:
    components: list[ConformalComponent]
    null_probability: float = 0.0
    infinity_rho_scale: float = 0.05
    metadata: dict[str, object] = field(default_factory=dict)

    def score(self, point: torch.Tensor) -> torch.Tensor:
        """Unified energy score for bounded, multimodal, and unbounded components."""
        if point.shape[-1] != 2:
            raise ValueError("point must end in two Cartesian coordinates")
        radius = torch.linalg.vector_norm(point, dim=-1).clamp_min(1e-12)
        phi = torch.atan2(point[..., 1], point[..., 0])
        rho = 1.0 / radius
        energies = []
        for component in self.components:
            mean = point.new_tensor(component.mean)
            covariance = point.new_tensor(component.covariance)
            if component.kind in {"near", "near_unbounded"}:
                value = point - mean
            elif component.kind in {"far", "far_unbounded"}:
                value = torch.stack([_wrap_angle(phi - mean[0]), rho - mean[1]], dim=-1)
            elif component.kind == "infinity":
                angular = _wrap_angle(phi - mean[0])
                value = torch.stack([angular, rho], dim=-1)
            else:
                raise ValueError(f"unknown component kind: {component.kind}")
            inverse = (
                point.new_tensor(component.precision)
                if component.precision is not None
                else torch.linalg.pinv(covariance, hermitian=True, rtol=1e-8)
            )
            mahalanobis = torch.einsum("...i,ij,...j->...", value, inverse, value)
            energies.append(
                math.log(max(component.weight, 1e-12)) - 0.5 * mahalanobis
            )
        if not energies:
            return point.new_full(point.shape[:-1], -math.log(max(self.null_probability, 1e-12)))
        stacked = torch.stack(energies, dim=-1)
        return -torch.logsumexp(stacked, dim=-1)

    def contains(self, point: torch.Tensor, threshold: float) -> torch.Tensor:
        if joint_state_decision(joint_state_probabilities(self)) is PithState.NULL:
            return torch.ones(point.shape[:-1], device=point.device, dtype=torch.bool)
        return self.score(point) <= threshold


def joint_state_probabilities(prediction: MixturePrediction) -> list[float]:
    """Return the post-refinement state mass in NEAR/FAR/INFINITY/NULL order.

    Geometry-mode weights already include the learned non-null state prior and the
    arc likelihood/selection score.  ``prediction_from_posteriors`` additionally
    moves invalid-Laplace mass into NULL.  Reporting this distribution separately
    from the raw CNN prior prevents contradictory state decisions.
    """
    mass = [0.0, 0.0, 0.0, float(prediction.null_probability)]
    mapping = {
        "near": PithState.NEAR,
        "near_unbounded": PithState.NEAR,
        "far": PithState.FAR,
        "far_unbounded": PithState.FAR,
        "infinity": PithState.INFINITY,
    }
    for component in prediction.components:
        mass[int(mapping[component.kind])] += float(component.weight)
    total = sum(mass)
    if not math.isfinite(total) or total <= 0:
        return [0.0, 0.0, 0.0, 1.0]
    return [value / total for value in mass]


def joint_state_decision(probabilities: list[float]) -> PithState:
    """MAP state with a conservative NULL tie-break, shared by every consumer."""
    if len(probabilities) != len(PithState):
        raise ValueError("joint state probabilities must contain four values")
    values = torch.tensor(probabilities, dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("joint state probabilities must be finite and nonnegative")
    if float(values.sum()) <= 0:
        return PithState.NULL
    if float(values[int(PithState.NULL)]) >= float(values[: int(PithState.NULL)].max()):
        return PithState.NULL
    return PithState(int(torch.argmax(values[: int(PithState.NULL)])))


def prediction_from_posteriors(
    posteriors: list[ModePosterior],
    *,
    infinity_rho_scale: float = 0.05,
    minimum_variance: float = 1e-6,
    null_probability: float = 0.0,
) -> MixturePrediction:
    components: list[ConformalComponent] = []
    unresolved_mass = 0.0
    for posterior in posteriors:
        result = posterior.result
        if not posterior.data.valid or not posterior.posterior.valid:
            # Invalid profile curvature is a failed local Laplace approximation,
            # not legitimate zero information.  The state/null branch must carry
            # this case instead of a zero-precision component.
            unresolved_mass += posterior.weight
            continue
        covariance = posterior.posterior.covariance
        data_full_rank = (
            posterior.data.rank == posterior.data.dimension
            and posterior.data.positive_semidefinite
        )
        if not data_full_rank:
            # The topology of the set is determined by image data, not by a prior
            # that happens to fill a zero-information direction.
            precision = _observable_precision(posterior.data.information)
            if precision is None:
                # A materially indefinite local Hessian is not a valid Laplace
                # component and must not create a hyperbolic "confidence" set.
                unresolved_mass += posterior.weight
                continue
            if posterior.data.rank == 0:
                unresolved_mass += posterior.weight
                continue
            if result.mode is RefinementMode.NEAR and result.center is not None:
                components.append(
                    ConformalComponent(
                        "near_unbounded",
                        posterior.weight,
                        result.center.detach().cpu().tolist(),
                        [[0.0, 0.0], [0.0, 0.0]],
                        precision.detach().cpu().tolist(),
                    )
                )
            elif result.mode is RefinementMode.FAR and result.direction is not None:
                phi = float(torch.atan2(result.direction[1], result.direction[0]))
                components.append(
                    ConformalComponent(
                        "far_unbounded",
                        posterior.weight,
                        [phi, result.inverse_distance or 0.0],
                        [[0.0, 0.0], [0.0, 0.0]],
                        precision.detach().cpu().tolist(),
                    )
                )
            elif result.direction is not None:
                phi = float(torch.atan2(result.direction[1], result.direction[0]))
                components.append(
                    ConformalComponent(
                        "infinity",
                        posterior.weight,
                        [phi, 0.0],
                        [[(math.pi / 2) ** 2, 0.0], [0.0, infinity_rho_scale**2]],
                    )
                )
            continue
        if covariance is None:
            # Full data rank but an invalid posterior Hessian: omit the local
            # Gaussian component instead of manufacturing a ridge covariance.
            unresolved_mass += posterior.weight
            continue
        covariance = covariance.detach().cpu()
        if result.mode is RefinementMode.NEAR and result.center is not None:
            components.append(
                ConformalComponent(
                    "near",
                    posterior.weight,
                    result.center.detach().cpu().tolist(),
                    (covariance + torch.eye(2) * minimum_variance).tolist(),
                )
            )
        elif result.mode is RefinementMode.FAR and result.direction is not None:
            phi = float(torch.atan2(result.direction[1], result.direction[0]))
            components.append(
                ConformalComponent(
                    "far",
                    posterior.weight,
                    [phi, result.inverse_distance or 0.0],
                    (covariance + torch.eye(2) * minimum_variance).tolist(),
                )
            )
        elif result.mode is RefinementMode.INFINITY and result.direction is not None:
            phi = float(torch.atan2(result.direction[1], result.direction[0]))
            angular_variance = max(float(covariance[0, 0]), minimum_variance)
            components.append(
                ConformalComponent(
                    "infinity",
                    posterior.weight,
                    [phi, 0.0],
                    [[angular_variance, 0.0], [0.0, infinity_rho_scale**2]],
                )
            )
    effective_null_probability = min(
        1.0, null_probability + (1.0 - null_probability) * unresolved_mass
    )
    total = sum(component.weight for component in components)
    if total > 0:
        for component in components:
            component.weight = (
                component.weight / total * max(0.0, 1.0 - effective_null_probability)
            )
    return MixturePrediction(
        components=components,
        null_probability=effective_null_probability,
        infinity_rho_scale=infinity_rho_scale,
    )


@dataclass
class ConformalCalibrator:
    alpha: float = 0.05
    threshold: float | None = None
    number_groups: int = 0
    reduction: str = "max"

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0,1)")
        if self.reduction not in {"max", "first"}:
            raise ValueError("reduction must be 'max' or 'first'")

    def fit(
        self,
        predictions: Iterable[MixturePrediction],
        truth: Iterable[torch.Tensor],
        group_ids: Iterable[str],
    ) -> "ConformalCalibrator":
        prediction_values = list(predictions)
        truth_values = list(truth)
        group_values = list(group_ids)
        if not (
            len(prediction_values) == len(truth_values) == len(group_values)
        ):
            raise ValueError("predictions, truth, and group_ids must have equal length")
        grouped: dict[str, list[float]] = {}
        for prediction, point, group in zip(
            prediction_values, truth_values, group_values, strict=True
        ):
            grouped.setdefault(group, []).append(float(prediction.score(point)))
        if not grouped:
            raise ValueError("calibration data are empty")
        if self.reduction == "max":
            scores = torch.tensor([max(values) for values in grouped.values()], dtype=torch.float64)
        else:
            scores = torch.tensor([values[0] for values in grouped.values()], dtype=torch.float64)
        self.number_groups = scores.numel()
        rank = math.ceil((self.number_groups + 1) * (1.0 - self.alpha))
        if rank > self.number_groups:
            # The exact finite-sample conformal quantile includes an infinity atom.
            # Replacing k=n+1 by the maximum observed score would under-cover.
            self.threshold = math.inf
        else:
            rank = max(rank, 1)
            self.threshold = float(torch.sort(scores).values[rank - 1])
        return self

    def contains(self, prediction: MixturePrediction, point: torch.Tensor) -> torch.Tensor:
        if self.threshold is None:
            raise RuntimeError("calibrator has not been fit")
        return prediction.contains(point, self.threshold)

    def save(self, path: str | Path) -> None:
        if self.threshold is None:
            raise RuntimeError("calibrator has not been fit")
        value = asdict(self)
        infinite = math.isinf(self.threshold)
        value["threshold"] = None if infinite else self.threshold
        value["threshold_is_infinite"] = infinite
        Path(path).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConformalCalibrator":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        infinite = bool(value.pop("threshold_is_infinite", False))
        if infinite:
            value["threshold"] = math.inf
        return cls(**value)
