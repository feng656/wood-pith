from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def _near_moments(output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.softmax(output["near_logits"], dim=-1)
    mean = (weight[..., None] * output["near_mean"]).sum(dim=1)
    component_cov = output["near_cholesky"] @ output["near_cholesky"].transpose(-1, -2)
    offset = output["near_mean"] - mean[:, None, :]
    covariance = (
        weight[..., None, None]
        * (component_cov + offset[..., :, None] * offset[..., None, :])
    ).sum(dim=1)
    return mean, covariance


def _sinkhorn_cost(
    cost: torch.Tensor,
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    entropy: float = 0.1,
    iterations: int = 20,
) -> torch.Tensor:
    log_kernel = -cost / entropy
    log_source = torch.log(source_weight.clamp_min(1e-8))
    log_target = torch.log(target_weight.clamp_min(1e-8))
    u = torch.zeros_like(log_source)
    v = torch.zeros_like(log_target)
    for _ in range(iterations):
        u = log_source - torch.logsumexp(log_kernel + v[:, None, :], dim=2)
        v = log_target - torch.logsumexp(log_kernel + u[:, :, None], dim=1)
    transport = torch.exp(log_kernel + u[:, :, None] + v[:, None, :])
    return (transport * cost).sum(dim=(1, 2)).mean()


def c4_equivariance_loss(
    original: dict[str, torch.Tensor], rotated: dict[str, torch.Tensor], quarter_turns: int
) -> dict[str, torch.Tensor]:
    """Consistency for an independently evaluated 90-degree rotation.

    The rotated copy is correlated augmentation evidence and is never fused as an
    independent posterior observation.
    """
    k = quarter_turns % 4
    if k == 0:
        raise ValueError("use a non-zero quarter turn for consistency training")
    expected_ring = torch.rot90(torch.sigmoid(original["ring_logits"]), k, (-2, -1))
    dense = functional.mse_loss(torch.sigmoid(rotated["ring_logits"]), expected_ring)
    expected_distance = torch.rot90(original["distance"], k, (-2, -1))
    dense = dense + functional.mse_loss(rotated["distance"], expected_distance)
    dense = dense + 0.25 * functional.mse_loss(
        rotated["log_variance"], torch.rot90(original["log_variance"], k, (-2, -1))
    )
    dense = dense + 0.25 * functional.mse_loss(
        rotated["defect_logits"], torch.rot90(original["defect_logits"], k, (-2, -1))
    )
    expected_orientation = torch.rot90(original["orientation"], k, (-2, -1))
    # An unoriented tangent uses doubled angle; a 90-degree turn negates both channels.
    if k % 2:
        expected_orientation = -expected_orientation
    dense = dense + functional.mse_loss(rotated["orientation"], expected_orientation)

    # torch.rot90 is counter-clockwise in array coordinates.  Our continuous
    # chart has x right and y down, hence the corresponding geometric angle is
    # negative.
    angle = -k * math.pi / 2.0
    rotation = original["near_mean"].new_tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    mean_a, covariance_a = _near_moments(original)
    mean_b, covariance_b = _near_moments(rotated)
    expected_mean = mean_a @ rotation.T
    expected_covariance = rotation @ covariance_a @ rotation.T
    global_loss = functional.mse_loss(mean_b, expected_mean) + functional.mse_loss(
        covariance_b, expected_covariance
    )
    global_loss = global_loss + functional.mse_loss(
        torch.softmax(rotated["state_logits"], -1),
        torch.softmax(original["state_logits"], -1),
    )
    far_a = (
        torch.softmax(original["far_logits"], -1)[..., None] * original["far_direction"]
    ).sum(1)
    far_b = (
        torch.softmax(rotated["far_logits"], -1)[..., None] * rotated["far_direction"]
    ).sum(1)
    global_loss = global_loss + functional.mse_loss(far_b, far_a @ rotation.T)
    # Component-level optimal transport prevents two different multimodal posteriors
    # from passing the test merely because their first two moments happen to match.
    near_cov_a = original["near_cholesky"] @ original["near_cholesky"].transpose(-1, -2)
    near_cov_a = rotation @ near_cov_a @ rotation.T
    near_cov_b = rotated["near_cholesky"] @ rotated["near_cholesky"].transpose(-1, -2)
    expected_components = original["near_mean"] @ rotation.T
    mean_cost = (expected_components[:, :, None, :] - rotated["near_mean"][:, None, :, :]).square().sum(-1)
    covariance_cost = (
        near_cov_a[:, :, None, :, :] - near_cov_b[:, None, :, :, :]
    ).square().mean(dim=(-1, -2))
    global_loss = global_loss + _sinkhorn_cost(
        mean_cost + 0.25 * covariance_cost,
        torch.softmax(original["near_logits"], -1),
        torch.softmax(rotated["near_logits"], -1),
    )
    expected_far_direction = original["far_direction"] @ rotation.T
    angular_cost = 1.0 - torch.einsum(
        "bik,bjk->bij", expected_far_direction, rotated["far_direction"]
    ).clamp(-1, 1)
    radial_cost = (
        original["far_log_eta_mean"][:, :, None]
        - rotated["far_log_eta_mean"][:, None, :]
    ).square()
    scale_cost = (
        torch.log(original["far_log_eta_scale"][:, :, None])
        - torch.log(rotated["far_log_eta_scale"][:, None, :])
    ).square()
    concentration_cost = (
        torch.log(original["far_kappa"][:, :, None])
        - torch.log(rotated["far_kappa"][:, None, :])
    ).square()
    global_loss = global_loss + _sinkhorn_cost(
        angular_cost + radial_cost + 0.1 * (scale_cost + concentration_cost),
        torch.softmax(original["far_logits"], -1),
        torch.softmax(rotated["far_logits"], -1),
    )
    global_loss = global_loss + functional.mse_loss(
        rotated["infinity_direction"], original["infinity_direction"] @ rotation.T
    )
    total = dense + global_loss
    return {"equivariance_total": total, "equivariance_dense": dense, "equivariance_global": global_loss}
