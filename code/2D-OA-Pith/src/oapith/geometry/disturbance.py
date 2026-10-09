from __future__ import annotations

import math

import torch
from torch import nn


class LatentMorphologyField(nn.Module):
    """Sparse inverse morphology corrections whose sources may lie outside the FOV.

    This is deliberately a statistical image-plane deformation, not a claim about
    a 3-D knot axis or a physical stress/strain field.
    """

    def __init__(
        self,
        number_sources: int,
        number_rings: int,
        reference_points: torch.Tensor,
        reference_ring_index: torch.Tensor,
        ring_order: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.number_sources = int(number_sources)
        self.number_rings = int(number_rings)
        self.register_buffer("reference_points", reference_points.detach().clone())
        self.register_buffer("reference_ring_index", reference_ring_index.detach().clone())
        resolved_order = (
            torch.arange(number_rings, device=reference_points.device)
            if ring_order is None
            else ring_order.to(device=reference_points.device, dtype=torch.long)
        )
        if resolved_order.numel() != number_rings:
            raise ValueError("ring_order must contain every ring exactly once")
        if not torch.equal(
            torch.sort(resolved_order).values,
            torch.arange(number_rings, device=reference_points.device),
        ):
            raise ValueError("ring_order must be a permutation of 0..number_rings-1")
        self.register_buffer("ring_order", resolved_order.detach().clone())
        if self.number_sources < 0:
            raise ValueError("number_sources must be nonnegative")
        if self.number_sources == 0:
            self.register_buffer("empty", reference_points.new_zeros(0))
            return
        center = reference_points.mean(0)
        extent = reference_points.std(0).clamp_min(0.25)
        angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            self.number_sources + 1,
            device=reference_points.device,
            dtype=reference_points.dtype,
        )[:-1]
        starts = center + 1.5 * extent * torch.stack([torch.cos(angles), torch.sin(angles)], -1)
        self.source_position = nn.Parameter(starts)
        self.log_scale = nn.Parameter(reference_points.new_zeros(self.number_sources, 2))
        self.source_angle = nn.Parameter(reference_points.new_zeros(self.number_sources))
        initial_vector = 1e-3 * torch.stack([torch.cos(angles), torch.sin(angles)], -1)
        self.vector = nn.Parameter(initial_vector)
        self.ring_amplitude = nn.Parameter(
            reference_points.new_full(
                (self.number_sources, self.number_rings), math.atanh(0.5)
            )
        )
        design = torch.cat(
            [reference_points.new_ones(reference_points.shape[0], 1), reference_points], dim=1
        )
        # Fixed reference geometry makes this projection differentiable with
        # respect to source parameters while removing the exact constant/affine
        # gauge shared with pith translation and the common shape matrix.
        self.register_buffer("reference_affine_design_pinv", torch.linalg.pinv(design))

    def forward(self, points: torch.Tensor, ring_index: torch.Tensor) -> torch.Tensor:
        field, _ = self.field_and_jacobian(points, ring_index)
        return field

    def _raw_field_and_jacobian(
        self, points: torch.Tensor, ring_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.number_sources == 0:
            return torch.zeros_like(points), points.new_zeros(points.shape[0], 2, 2)
        delta = points[:, None, :] - self.source_position[None, :, :]
        cosine, sine = torch.cos(self.source_angle), torch.sin(self.source_angle)
        rotation = torch.stack(
            [torch.stack([cosine, sine], -1), torch.stack([-sine, cosine], -1)], -2
        )
        local = torch.einsum("nsj,sjk->nsk", delta, rotation)
        scale = torch.exp(self.log_scale).clamp(0.05, 8.0)
        kernel = torch.exp(-0.5 * (local / scale[None, :, :]).square().sum(-1))
        # Signed, smooth ring-specific effects may vanish or reverse.  The source
        # vector carries the base direction; the bounded amplitude prevents an
        # unconstrained scale exchange from exploding.
        amplitude = torch.tanh(self.ring_amplitude[:, ring_index].T)
        weight = kernel * amplitude
        field = weight[..., None].mul(self.vector[None, :, :]).sum(1)
        gradient_log_kernel = -torch.einsum(
            "nsk,skj->nsj", local / scale[None, :, :].square(), rotation.transpose(-1, -2)
        )
        jacobian = torch.einsum("ns,so,nsj->noj", weight, self.vector, gradient_log_kernel)
        return field, jacobian

    def field_and_jacobian(
        self, points: torch.Tensor, ring_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.number_sources == 0:
            return torch.zeros_like(points), points.new_zeros(points.shape[0], 2, 2)
        raw_field, raw_jacobian = self._raw_field_and_jacobian(points, ring_index)
        reference_field, _ = self._raw_field_and_jacobian(
            self.reference_points, self.reference_ring_index
        )
        coefficients = self.reference_affine_design_pinv @ reference_field
        design = torch.cat([points.new_ones(points.shape[0], 1), points], dim=1)
        field = raw_field - design @ coefficients
        affine_jacobian = coefficients[1:].T
        jacobian = raw_jacobian - affine_jacobian[None]
        return field, jacobian

    def regularization(self) -> torch.Tensor:
        if self.number_sources == 0:
            return self.empty.sum()
        ordered_amplitude = self.ring_amplitude[:, self.ring_order]
        amplitude_smoothness = (
            (ordered_amplitude[:, 1:] - ordered_amplitude[:, :-1]).square().sum()
            if self.number_rings > 1
            else self.ring_amplitude.new_zeros(())
        )
        scale_prior = self.log_scale.square().sum()
        _, jacobian = self.field_and_jacobian(
            self.reference_points, self.reference_ring_index
        )
        # ||J||_2 <= ||J||_F < 0.75 implies sigma_min(I-J) > 0.25.  Penalizing the
        # squared Frobenius bound avoids unstable second derivatives of repeated
        # singular values when the observed Hessian is computed.
        jacobian_energy = jacobian.square().sum(dim=(-1, -2))
        nonfolding = 10.0 * torch.nn.functional.relu(
            jacobian_energy - 0.75**2
        ).square().mean()
        excessive_gradient = torch.nn.functional.relu(
            jacobian_energy - 0.60**2
        ).square().mean()
        return (
            2.0 * self.vector.square().sum()
            + 5e-2 * amplitude_smoothness
            + 2e-2 * self.ring_amplitude.square().sum()
            + 1e-2 * scale_prior
            + nonfolding
            + excessive_gradient
        )

    def source_summary(self) -> list[dict[str, list[float] | float]]:
        if self.number_sources == 0:
            return []
        return [
            {
                "position": self.source_position[index].detach().cpu().tolist(),
                "scale": torch.exp(self.log_scale[index]).detach().cpu().tolist(),
                "angle": float(self.source_angle[index].detach()),
                "vector": self.vector[index].detach().cpu().tolist(),
            }
            for index in range(self.number_sources)
        ]
