from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct, DecoderBlock, ResidualBlock


class OAPithNet(nn.Module):
    """Dense ring evidence plus a finite/far/infinity/null probabilistic head."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        embedding_dim: int = 8,
        defect_classes: int = 4,
        mixture_components: int = 4,
        near_extent: float = 2.5,
        min_scale: float = 1e-3,
        detach_variance_features: bool = True,
    ) -> None:
        super().__init__()
        integer_values = {
            "base_channels": base_channels,
            "embedding_dim": embedding_dim,
            "defect_classes": defect_classes,
            "mixture_components": mixture_components,
        }
        if any(not isinstance(value, int) or value < 1 for value in integer_values.values()):
            raise ValueError(f"model dimensions must be positive integers: {integer_values}")
        if near_extent <= 0 or min_scale <= 0:
            raise ValueError("near_extent and min_scale must be positive")
        if not isinstance(detach_variance_features, bool):
            raise ValueError("detach_variance_features must be boolean")
        self.base_channels = base_channels
        self.mixture_components = mixture_components
        self.near_extent = float(near_extent)
        self.min_scale = float(min_scale)
        self.detach_variance_features = bool(detach_variance_features)
        channels = [base_channels * factor for factor in (1, 2, 4, 8, 12)]
        self.stem = nn.Sequential(
            ConvNormAct(3, channels[0]),
            ResidualBlock(channels[0], channels[0]),
        )
        self.encoder = nn.ModuleList(
            [
                ResidualBlock(channels[0], channels[1], 2),
                ResidualBlock(channels[1], channels[2], 2),
                ResidualBlock(channels[2], channels[3], 2),
                ResidualBlock(channels[3], channels[4], 2),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(channels[4], channels[3], channels[3]),
                DecoderBlock(channels[3], channels[2], channels[2]),
                DecoderBlock(channels[2], channels[1], channels[1]),
                DecoderBlock(channels[1], channels[0], channels[0]),
            ]
        )
        # Mean/topology tasks share a decoder. Heteroscedastic variance has a
        # separate head and, by default, sees a detached representation so its NLL
        # cannot modify the shared mean predictor.
        dense_channels = 1 + 1 + 2 + embedding_dim + defect_classes
        self.dense_head = nn.Sequential(
            ConvNormAct(channels[0], channels[0]),
            nn.Conv2d(channels[0], dense_channels, 1),
        )
        self.variance_head = nn.Sequential(
            ConvNormAct(channels[0], channels[0]),
            nn.Conv2d(channels[0], 1, 1),
        )
        self.embedding_dim = embedding_dim
        self.defect_classes = defect_classes

        k = mixture_components
        global_width = channels[4]
        output_width = 4 + k + 2 * k + 3 * k + k + 2 * k + k + k + k + 2 + 1
        self.global_head = nn.Sequential(
            nn.Linear(global_width, global_width),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(global_width, output_width),
        )

    def _masked_pool(
        self, feature: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if valid_mask is None:
            return feature.mean(dim=(-2, -1))
        mask = torch.nn.functional.interpolate(valid_mask, size=feature.shape[-2:], mode="area")
        numerator = (feature * mask).sum(dim=(-2, -1))
        return numerator / mask.sum(dim=(-2, -1)).clamp_min(1e-6)

    def _decode_global(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        k = self.mixture_components
        cursor = 0

        def take(width: int) -> torch.Tensor:
            nonlocal cursor
            value = raw[:, cursor : cursor + width]
            cursor += width
            return value

        state_logits = take(4)
        near_logits = take(k)
        near_mean = torch.tanh(take(2 * k).view(-1, k, 2)) * self.near_extent
        chol_raw = take(3 * k).view(-1, k, 3)
        diagonal = torch.nn.functional.softplus(chol_raw[..., [0, 2]]) + self.min_scale
        near_cholesky = torch.zeros(
            raw.shape[0], k, 2, 2, device=raw.device, dtype=raw.dtype
        )
        near_cholesky[..., 0, 0] = diagonal[..., 0]
        near_cholesky[..., 1, 0] = chol_raw[..., 1]
        near_cholesky[..., 1, 1] = diagonal[..., 1]

        far_logits = take(k)
        far_direction = torch.nn.functional.normalize(
            take(2 * k).view(-1, k, 2), dim=-1, eps=1e-8
        )
        far_log_eta_mean = -torch.nn.functional.softplus(take(k))
        far_log_eta_scale = torch.nn.functional.softplus(take(k)) + self.min_scale
        far_kappa = torch.nn.functional.softplus(take(k)).clamp_max(1e3) + self.min_scale
        infinity_direction = torch.nn.functional.normalize(take(2), dim=-1, eps=1e-8)
        infinity_kappa = torch.nn.functional.softplus(take(1)).squeeze(-1) + self.min_scale
        if cursor != raw.shape[1]:
            raise RuntimeError("global head layout is inconsistent")
        return {
            "state_logits": state_logits,
            "near_logits": near_logits,
            "near_mean": near_mean,
            "near_cholesky": near_cholesky,
            "far_logits": far_logits,
            "far_direction": far_direction,
            "far_log_eta_mean": far_log_eta_mean,
            "far_log_eta_scale": far_log_eta_scale,
            "far_kappa": far_kappa,
            "infinity_direction": infinity_direction,
            "infinity_kappa": infinity_kappa,
        }

    def forward(
        self, image: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        features = [self.stem(image)]
        for block in self.encoder:
            features.append(block(features[-1]))
        decoded = features[-1]
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            decoded = block(decoded, skip)
        raw_dense = self.dense_head(decoded)
        cursor = 0
        ring_logits = raw_dense[:, cursor : cursor + 1]
        cursor += 1
        distance = torch.sigmoid(raw_dense[:, cursor : cursor + 1])
        cursor += 1
        variance_features = decoded.detach() if self.detach_variance_features else decoded
        log_variance = self.variance_head(variance_features).clamp(-8.0, 6.0)
        orientation = torch.nn.functional.normalize(
            raw_dense[:, cursor : cursor + 2], dim=1, eps=1e-8
        )
        cursor += 2
        embedding = torch.nn.functional.normalize(
            raw_dense[:, cursor : cursor + self.embedding_dim], dim=1, eps=1e-8
        )
        cursor += self.embedding_dim
        defect_logits = raw_dense[:, cursor : cursor + self.defect_classes]
        cursor += self.defect_classes
        if cursor != raw_dense.shape[1]:
            raise RuntimeError("dense head layout is inconsistent")
        global_raw = self.global_head(self._masked_pool(features[-1], valid_mask))
        return {
            "ring_logits": ring_logits,
            "distance": distance,
            "log_variance": log_variance,
            "orientation": orientation,
            "embedding": embedding,
            "defect_logits": defect_logits,
            **self._decode_global(global_raw),
        }
