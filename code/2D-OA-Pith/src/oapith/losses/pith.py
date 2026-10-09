from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from oapith.geometry.coordinates import cartesian_to_far
from oapith.types import PithState


def _log_i0(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.special.i0e(value).clamp_min(1e-30)) + value.abs()


def circular_log_probability(
    predicted_direction: torch.Tensor, target_direction: torch.Tensor, kappa: torch.Tensor
) -> torch.Tensor:
    dot = (predicted_direction * target_direction.unsqueeze(-2)).sum(dim=-1)
    return kappa * dot - math.log(2.0 * math.pi) - _log_i0(kappa)


def near_mixture_nll(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_covariance: torch.Tensor | None = None,
) -> torch.Tensor:
    difference = target[:, None, :] - output["near_mean"]
    covariance = output["near_cholesky"] @ output["near_cholesky"].transpose(-1, -2)
    if target_covariance is not None:
        covariance = covariance + target_covariance[:, None, :, :]
    cholesky = torch.linalg.cholesky(covariance + torch.eye(2, device=target.device) * 1e-7)
    solved = torch.cholesky_solve(difference.unsqueeze(-1), cholesky).squeeze(-1)
    mahalanobis = (difference * solved).sum(-1)
    logdet = 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)
    component = -0.5 * (mahalanobis + logdet + 2.0 * math.log(2.0 * math.pi))
    mixture = torch.log_softmax(output["near_logits"], dim=-1)
    return -torch.logsumexp(mixture + component, dim=-1)


def far_mixture_nll(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    switch_radius: float,
    target_covariance: torch.Tensor | None = None,
) -> torch.Tensor:
    direction, eta = cartesian_to_far(target, switch_radius)
    log_eta = torch.log(eta.clamp_min(1e-8)).unsqueeze(-1)
    eta_scale = output["far_log_eta_scale"]
    if target_covariance is None:
        radial = -0.5 * ((log_eta - output["far_log_eta_mean"]) / eta_scale).square()
        radial = radial - torch.log(eta_scale) - 0.5 * math.log(2.0 * math.pi)
        angular = circular_log_probability(
            output["far_direction"], direction, output["far_kappa"]
        )
        component = radial + angular
    else:
        # Delta-method propagation from Cartesian (x,y) into the far chart
        # (phi, log eta).  The off-diagonal term matters for an elongated pith
        # registration covariance and is retained in the 2-D component NLL.
        x, y = target.unbind(-1)
        radius_squared = target.square().sum(-1).clamp_min(1e-8)
        jacobian = torch.stack(
            [
                torch.stack([-y / radius_squared, x / radius_squared], -1),
                torch.stack([-x / radius_squared, -y / radius_squared], -1),
            ],
            dim=-2,
        )
        chart_covariance = jacobian @ target_covariance @ jacobian.transpose(-1, -2)
        predicted_covariance = torch.diag_embed(
            torch.stack(
                [output["far_kappa"].reciprocal(), eta_scale.square()], dim=-1
            )
        )
        covariance = predicted_covariance + chart_covariance[:, None]
        covariance = covariance + torch.eye(2, device=target.device, dtype=target.dtype) * 1e-7
        angular_residual = torch.atan2(
            output["far_direction"][..., 0] * direction[:, None, 1]
            - output["far_direction"][..., 1] * direction[:, None, 0],
            (output["far_direction"] * direction[:, None]).sum(-1),
        )
        residual = torch.stack(
            [angular_residual, log_eta - output["far_log_eta_mean"]], dim=-1
        )
        cholesky = torch.linalg.cholesky(covariance)
        solved = torch.cholesky_solve(residual.unsqueeze(-1), cholesky).squeeze(-1)
        mahalanobis = (residual * solved).sum(-1)
        logdet = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(-1)
        component = -0.5 * (mahalanobis + logdet + 2.0 * math.log(2.0 * math.pi))
    mixture = torch.log_softmax(output["far_logits"], dim=-1)
    return -torch.logsumexp(mixture + component, dim=-1)


def probabilistic_pith_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_covariance: torch.Tensor,
    target_valid: torch.Tensor,
    state_target: torch.Tensor,
    *,
    switch_radius: float = 2.0,
    state_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    state_known = state_target >= 0
    state = (
        functional.cross_entropy(output["state_logits"][state_known], state_target[state_known])
        if state_known.any()
        else output["state_logits"].sum() * 0.0
    )
    inferred_state = torch.where(
        torch.linalg.vector_norm(target, dim=-1) <= switch_radius,
        torch.full_like(state_target, int(PithState.NEAR)),
        torch.full_like(state_target, int(PithState.FAR)),
    )
    effective_state = torch.where(state_known, state_target, inferred_state)
    near = near_mixture_nll(output, target, target_covariance)
    far = far_mixture_nll(output, target, switch_radius, target_covariance)
    direction = functional.normalize(target, dim=-1, eps=1e-8)
    infinity = -circular_log_probability(
        output["infinity_direction"].unsqueeze(1),
        direction,
        output["infinity_kappa"].unsqueeze(1),
    ).squeeze(1)
    localization = torch.where(
        effective_state == int(PithState.NEAR),
        near,
        torch.where(effective_state == int(PithState.FAR), far, infinity),
    )
    informative = target_valid & (effective_state != int(PithState.NULL))
    localization_loss = (
        localization[informative].mean()
        if informative.any()
        else output["state_logits"].sum() * 0.0
    )
    total = localization_loss + state_weight * state
    return {
        "pith_total": total,
        "pith_localization_nll": localization_loss,
        "pith_state_ce": state,
    }
