from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from oapith.inference.refiner import (
    PackedArcs,
    RefinementResult,
    _ProbabilisticLayerModel,
)


@dataclass
class InformationDiagnostics:
    information: torch.Tensor
    covariance: torch.Tensor | None
    rank: int
    dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    pseudo_logdet: float
    nuisance_rank: int
    nuisance_dimension: int
    positive_semidefinite: bool
    valid: bool
    invalid_reason: str | None


@dataclass
class ModePosterior:
    result: RefinementResult
    weight: float
    data: InformationDiagnostics
    posterior: InformationDiagnostics


def _invalid_information(
    dimension: int, like: torch.Tensor, reason: str
) -> InformationDiagnostics:
    return InformationDiagnostics(
        information=like.new_zeros(dimension, dimension),
        covariance=None,
        rank=0,
        dimension=dimension,
        minimum_eigenvalue=math.nan,
        maximum_eigenvalue=math.nan,
        condition_number=math.inf,
        pseudo_logdet=-math.inf,
        nuisance_rank=0,
        nuisance_dimension=max(0, like.shape[0] - dimension),
        positive_semidefinite=False,
        valid=False,
        invalid_reason=reason,
    )


def _flatten_parameters(
    model: torch.nn.Module,
) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int]]]:
    values, layout = [], []
    parameters = [item for item in model.named_parameters() if item[1].requires_grad]
    parameters.sort(key=lambda item: (item[0] != "target", item[0]))
    for name, parameter in parameters:
        flat = parameter.detach().reshape(-1)
        values.append(flat)
        layout.append((name, parameter.shape, flat.numel()))
    return torch.cat(values).requires_grad_(True), layout


def _unflatten(vector: torch.Tensor, layout: list[tuple[str, torch.Size, int]]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    cursor = 0
    for name, shape, count in layout:
        result[name] = vector[cursor : cursor + count].reshape(shape)
        cursor += count
    if cursor != vector.numel():
        raise RuntimeError("parameter layout does not cover the vector")
    return result


def _scaled_stationarity(
    parameters: list[torch.Tensor],
    gradient: tuple[torch.Tensor | None, ...],
    effective_sample_size: torch.Tensor,
) -> torch.Tensor:
    """Maximum natural-parameter gradient per effective observation.

    A stationarity diagnostic must be invariant to adding a constant to the
    objective.  Dividing by ``abs(objective)`` violates that requirement and can
    make an unfinished optimizer look stationary merely because the NLL contains
    a large constant.  The likelihood caps every arc at 24 independent samples,
    so the same effective count used for model selection is the appropriate scale
    for gradients of the summed objective.
    """
    scaled_terms = [
        (derivative.detach().abs() * parameter.detach().abs().clamp_min(1.0)).max()
        for parameter, derivative in zip(parameters, gradient, strict=True)
        # 0 元素参数（如单环近场模型的 gap 参数）没有 stationarity 可查，
        # 对空张量调用 .max() 会抛 "input.numel() == 0"，直接跳过。
        if derivative is not None and parameter.numel() > 0
    ]
    if not scaled_terms:
        return effective_sample_size.new_zeros(())
    return torch.stack(scaled_terms).max() / effective_sample_size.clamp_min(1.0)


def observed_hessian(
    model: _ProbabilisticLayerModel,
    data: PackedArcs,
    *,
    include_prior: bool,
) -> torch.Tensor:
    """Hessian of the marginalized arc-mixture likelihood, not frozen EM weights."""
    vector, layout = _flatten_parameters(model)

    def objective(value: torch.Tensor) -> torch.Tensor:
        parameters = _unflatten(value, layout)
        return torch.func.functional_call(
            model,
            parameters,
            (data,),
            {"reliability": None, "marginal": True, "include_prior": include_prior},
        )

    hessian = torch.autograd.functional.hessian(objective, vector, vectorize=True)
    return 0.5 * (hessian + hessian.T)


def marginalize_information(
    hessian: torch.Tensor,
    target_dimension: int,
    *,
    relative_tolerance: float = 1e-8,
) -> InformationDiagnostics:
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be square")
    if not 0 < target_dimension <= hessian.shape[0]:
        raise ValueError("invalid target dimension")
    if not bool(torch.isfinite(hessian).all()):
        return _invalid_information(target_dimension, hessian, "non-finite Hessian")
    target = hessian[:target_dimension, :target_dimension]
    cross = hessian[:target_dimension, target_dimension:]
    nuisance = hessian[target_dimension:, target_dimension:]
    if nuisance.numel():
        nuisance = 0.5 * (nuisance + nuisance.T)
        eigenvalues_u, eigenvectors_u = torch.linalg.eigh(nuisance)
        threshold_u = relative_tolerance * eigenvalues_u.abs().max().clamp_min(1.0)
        if bool((eigenvalues_u < -threshold_u).any()):
            return InformationDiagnostics(
                information=torch.zeros_like(target).detach(),
                covariance=None,
                rank=0,
                dimension=target_dimension,
                minimum_eigenvalue=float(eigenvalues_u.min()),
                maximum_eigenvalue=float(eigenvalues_u.max()),
                condition_number=math.inf,
                pseudo_logdet=-math.inf,
                nuisance_rank=int((eigenvalues_u > threshold_u).sum()),
                nuisance_dimension=nuisance.shape[0],
                positive_semidefinite=False,
                valid=False,
                invalid_reason="indefinite nuisance Hessian",
            )
        null_u = eigenvalues_u <= threshold_u
        if bool(null_u.any()):
            null_cross = cross @ eigenvectors_u[:, null_u]
            cross_tolerance = relative_tolerance * cross.abs().max().clamp_min(1.0)
            if bool((null_cross.abs() > cross_tolerance).any()):
                return InformationDiagnostics(
                    information=torch.zeros_like(target).detach(),
                    covariance=None,
                    rank=0,
                    dimension=target_dimension,
                    minimum_eigenvalue=float(eigenvalues_u.min()),
                    maximum_eigenvalue=float(eigenvalues_u.max()),
                    condition_number=math.inf,
                    pseudo_logdet=-math.inf,
                    nuisance_rank=int((eigenvalues_u > threshold_u).sum()),
                    nuisance_dimension=nuisance.shape[0],
                    positive_semidefinite=True,
                    valid=False,
                    invalid_reason="target couples to nuisance null space",
                )
        keep_u = eigenvalues_u > threshold_u
        nuisance_rank = int(keep_u.sum())
        if keep_u.any():
            projected = eigenvectors_u[:, keep_u]
            solved = projected @ (
                (projected.T @ cross.T) / eigenvalues_u[keep_u, None]
            )
            information = target - cross @ solved
        else:
            information = target.clone()
        nuisance_dimension = nuisance.shape[0]
    else:
        information = target.clone()
        nuisance_rank = nuisance_dimension = 0
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(information)
    threshold = relative_tolerance * eigenvalues.abs().max().clamp_min(1.0)
    positive = eigenvalues > threshold
    rank = int(positive.sum())
    covariance = None
    if rank == target_dimension and bool((eigenvalues > 0).all()):
        covariance = eigenvectors @ torch.diag(eigenvalues.reciprocal()) @ eigenvectors.T
    positive_values = eigenvalues[positive]
    pseudo_logdet = float(torch.log(positive_values).sum()) if positive_values.numel() else -math.inf
    minimum = float(eigenvalues.min())
    maximum = float(eigenvalues.max())
    condition = (
        float(positive_values.max() / positive_values.min())
        if positive_values.numel() == target_dimension
        else math.inf
    )
    is_psd = bool(minimum >= -float(threshold))
    return InformationDiagnostics(
        information=information.detach(),
        covariance=None if covariance is None else covariance.detach(),
        rank=rank,
        dimension=target_dimension,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        condition_number=condition,
        pseudo_logdet=pseudo_logdet,
        nuisance_rank=nuisance_rank,
        nuisance_dimension=nuisance_dimension,
        positive_semidefinite=is_psd,
        valid=is_psd,
        invalid_reason=None if is_psd else "indefinite marginalized target information",
    )


def build_mode_posteriors(
    results: list[RefinementResult],
    models: list[_ProbabilisticLayerModel],
    data: PackedArcs,
) -> list[ModePosterior]:
    if len(results) != len(models):
        raise ValueError("results and models must have equal length")
    log_weights = torch.tensor([result.evidence_score for result in results], dtype=torch.float64)
    weights = torch.softmax(log_weights, dim=0)
    posteriors = []
    for result, model, weight in zip(results, models, weights):
        target_dimension = model.target.numel()
        gradient_norm_value = math.nan
        scaled_stationarity_value = math.nan
        try:
            parameters = [value for value in model.parameters() if value.requires_grad]
            objective = model(data, marginal=True, include_prior=True)
            gradient = torch.autograd.grad(objective, parameters, allow_unused=True)
            gradient_terms = [
                value.square().sum() for value in gradient if value is not None
            ]
            gradient_norm = (
                torch.sqrt(torch.stack(gradient_terms).sum())
                if gradient_terms
                else objective.new_zeros(())
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite marginal gradient")
            effective_sample_size = torch.minimum(
                data.counts, data.counts.new_full(data.counts.shape, 24.0)
            ).sum()
            scaled_stationarity = _scaled_stationarity(
                parameters,
                gradient,
                effective_sample_size,
            )
            gradient_norm_value = float(gradient_norm)
            scaled_stationarity_value = float(scaled_stationarity)
            if not bool(torch.isfinite(scaled_stationarity)):
                raise FloatingPointError("non-finite scaled stationarity diagnostic")
            if float(scaled_stationarity) > model.config.stationarity_tolerance:
                raise FloatingPointError(
                    "non-stationary MAP: scaled gradient "
                    f"{float(scaled_stationarity):.3e} exceeds "
                    f"{model.config.stationarity_tolerance:.3e}"
                )
            data_hessian = observed_hessian(model, data, include_prior=False)
            posterior_hessian = observed_hessian(model, data, include_prior=True)
            data_information = marginalize_information(data_hessian, target_dimension)
            posterior_information = marginalize_information(posterior_hessian, target_dimension)
        except (RuntimeError, FloatingPointError, ValueError) as error:
            reason = f"local posterior construction failed: {type(error).__name__}: {error}"
            like = model.target.detach()
            data_information = _invalid_information(target_dimension, like, reason)
            posterior_information = _invalid_information(target_dimension, like, reason)
        result.diagnostics["marginal_gradient_norm"] = gradient_norm_value
        result.diagnostics["scaled_stationarity"] = scaled_stationarity_value
        result.diagnostics["stationarity_tolerance"] = model.config.stationarity_tolerance
        result.diagnostics["stationarity_effective_sample_size"] = float(
            torch.minimum(
                data.counts, data.counts.new_full(data.counts.shape, 24.0)
            ).sum()
        )
        result.diagnostics.update(
            {
                "data_rank": data_information.rank,
                "data_condition": data_information.condition_number,
                "data_information_valid": data_information.valid,
                "data_information_invalid_reason": data_information.invalid_reason or "",
                "posterior_rank": posterior_information.rank,
                "posterior_condition": posterior_information.condition_number,
                "posterior_information_valid": posterior_information.valid,
                "posterior_information_invalid_reason": posterior_information.invalid_reason or "",
                "prior_supported_only": data_information.valid
                and posterior_information.valid
                and data_information.rank < target_dimension
                and posterior_information.rank == target_dimension,
            }
        )
        posteriors.append(
            ModePosterior(
                result=result,
                weight=float(weight),
                data=data_information,
                posterior=posterior_information,
            )
        )
    return posteriors
