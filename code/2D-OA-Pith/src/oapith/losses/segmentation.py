from __future__ import annotations

import torch
import torch.nn.functional as functional


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    horizontal = -functional.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    vertical = -functional.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(horizontal, vertical)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return functional.max_pool2d(image, 3, stride=1, padding=1)


def soft_skeletonize(image: torch.Tensor, iterations: int = 30) -> torch.Tensor:
    opened = soft_dilate(soft_erode(image))
    skeleton = functional.relu(image - opened)
    current = image
    for _ in range(iterations):
        current = soft_erode(current)
        opened = soft_dilate(soft_erode(current))
        delta = functional.relu(current - opened)
        skeleton = skeleton + functional.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, iterations: int = 30
) -> torch.Tensor:
    if not bool((valid > 0).any()) or not bool(((target * valid) > 0).any()):
        return prediction.sum() * 0.0
    prediction = prediction * valid
    target = target * valid
    pred_skeleton = soft_skeletonize(prediction, iterations)
    target_skeleton = soft_skeletonize(target, iterations)
    smooth = 1e-6
    topology_precision = (pred_skeleton * target).sum() / pred_skeleton.sum().clamp_min(smooth)
    topology_sensitivity = (target_skeleton * prediction).sum() / target_skeleton.sum().clamp_min(
        smooth
    )
    cldice = 2.0 * topology_precision * topology_sensitivity / (
        topology_precision + topology_sensitivity
    ).clamp_min(smooth)
    return 1.0 - cldice


def discriminative_embedding_loss(
    embedding: torch.Tensor,
    ring_index: torch.Tensor,
    valid: torch.Tensor,
    *,
    delta_variance: float = 0.25,
    delta_distance: float = 0.75,
) -> torch.Tensor:
    losses = []
    for batch in range(embedding.shape[0]):
        labels = ring_index[batch]
        foreground = (labels >= 0) & (valid[batch, 0] > 0.5)
        unique = torch.unique(labels[foreground])
        if unique.numel() == 0:
            continue
        means = []
        variance = embedding.new_zeros(())
        for label in unique:
            mask = foreground & (labels == label)
            vectors = embedding[batch, :, mask].transpose(0, 1)
            mean = vectors.mean(dim=0)
            means.append(mean)
            distances = torch.linalg.vector_norm(vectors - mean, dim=-1)
            variance = variance + functional.relu(distances - delta_variance).square().mean()
        variance = variance / unique.numel()
        means_tensor = torch.stack(means)
        if len(means) > 1:
            pairwise = torch.cdist(means_tensor, means_tensor)
            eye = torch.eye(len(means), device=embedding.device, dtype=torch.bool)
            separation = functional.relu(2 * delta_distance - pairwise[~eye]).square().mean()
        else:
            separation = embedding.new_zeros(())
        regularization = torch.linalg.vector_norm(means_tensor, dim=-1).mean()
        losses.append(variance + separation + 1e-3 * regularization)
    return torch.stack(losses).mean() if losses else embedding.sum() * 0.0


def dense_ring_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    *,
    cldice_weight: float = 0.25,
    distance_weight: float = 0.25,
    orientation_weight: float = 0.20,
    embedding_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    valid = valid_mask.to(output["ring_logits"].dtype)
    soft_target = target["ring_probability"].to(output["ring_logits"].dtype)
    annotation_weight = target["label_weight"].to(output["ring_logits"].dtype)
    base_weight = valid * (0.25 + 0.75 * annotation_weight)
    pointwise_bce = functional.binary_cross_entropy_with_logits(
        output["ring_logits"], soft_target, reduction="none"
    )
    # Keep an unweighted mean-prediction path: uncertainty is not allowed to
    # suppress this gradient merely by inflating a variance estimate.  This is a
    # faithful decoupling when OAPithNet.detach_variance_features is left enabled.
    bce = _masked_mean(pointwise_bce, base_weight)
    probability = torch.sigmoid(output["ring_logits"])
    dice = 1.0 - (2 * (probability * soft_target * valid).sum() + 1e-6) / (
        (probability * valid).sum() + (soft_target * valid).sum() + 1e-6
    )
    cldice = soft_cldice_loss(probability, soft_target, valid)
    distance_residual = output["distance"] - target["distance"]
    # The SDF head needs global background context, but its uncertainty is consumed
    # specifically at the extracted ridge.  A boundary-focused continuous weight
    # prevents saturated distance=1 background pixels from overwhelming the normal
    # localization variance while retaining a smaller full-field SDF term.
    boundary_weight = torch.exp(-0.5 * (target["distance"] / 0.35).square())
    distance_supervision_weight = valid * (0.2 + 0.8 * boundary_weight)
    distance = _masked_mean(functional.smooth_l1_loss(
        output["distance"], target["distance"], reduction="none"
    ), distance_supervision_weight)
    predicted_variance = torch.exp(output["log_variance"])
    annotation_variance = target["annotation_sigma_distance"].square()
    total_variance = predicted_variance + annotation_variance
    variance_calibration = _masked_mean(
        0.5 * (distance_residual.detach().square() / total_variance + torch.log(total_variance)),
        valid * boundary_weight,
    )
    orientation_valid = target["orientation_valid"] * valid
    orientation_dot = (output["orientation"] * target["orientation"]).sum(dim=1, keepdim=True)
    orientation = _masked_mean(1.0 - orientation_dot.clamp(-1.0, 1.0), orientation_valid)
    embedding = discriminative_embedding_loss(
        output["embedding"], target["ring_index"], valid
    )
    total = (
        bce
        + dice
        + cldice_weight * cldice
        + distance_weight * (distance + 0.1 * variance_calibration)
        + orientation_weight * orientation
        + embedding_weight * embedding
    )
    return {
        "dense_total": total,
        "ring_soft_bce": bce,
        "ring_dice": dice,
        "ring_cldice": cldice,
        "distance": distance,
        "variance_calibration": variance_calibration,
        "orientation": orientation,
        "embedding": embedding,
    }
