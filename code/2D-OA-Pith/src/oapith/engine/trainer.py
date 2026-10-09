from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from oapith import CHECKPOINT_SCHEMA_VERSION, COORDINATE_CONTRACT
from oapith.contracts import (
    resolved_model_contract,
    resolved_preprocessing_contract,
    validate_checkpoint_contract,
)
from oapith.losses import c4_equivariance_loss, dense_ring_loss, probabilistic_pith_loss


@dataclass
class TrainerConfig:
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    pith_weight: float = 1.0
    equivariance_weight: float = 0.15
    gradient_clip: float = 5.0
    amp: bool = True
    switch_radius: float = 2.0
    seed: int = 2026
    checkpoint_every: int = 5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainerConfig,
        output_directory: str | Path,
        *,
        device: torch.device | str | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.run_config = run_config
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(config.epochs, 1)
        )
        amp_enabled = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.epoch = 0
        self.best_validation = float("inf")
        seed_everything(config.seed)

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        for key in (
            "image",
            "valid_mask",
            "pith",
            "pith_covariance",
            "pith_valid",
            "rings_annotated",
            "state_target",
        ):
            batch[key] = batch[key].to(self.device, non_blocking=True)
        batch["dense"] = {
            key: value.to(self.device, non_blocking=True) for key, value in batch["dense"].items()
        }
        return batch

    def _loss(self, batch: dict[str, Any], training: bool) -> dict[str, torch.Tensor]:
        image, valid = batch["image"], batch["valid_mask"]
        output = self.model(image, valid)
        annotation_gate = batch["rings_annotated"].to(valid.dtype)[:, None, None, None]
        dense = dense_ring_loss(output, batch["dense"], valid * annotation_gate)
        pith = probabilistic_pith_loss(
            output,
            batch["pith"],
            batch["pith_covariance"],
            batch["pith_valid"],
            batch["state_target"],
            switch_radius=self.config.switch_radius,
        )
        total = dense["dense_total"] + self.config.pith_weight * pith["pith_total"]
        metrics = {**dense, **pith}
        if training and self.config.equivariance_weight > 0:
            quarter_turns = int(torch.randint(1, 4, (), device=image.device))
            rotated_image = torch.rot90(image, quarter_turns, (-2, -1))
            rotated_valid = torch.rot90(valid, quarter_turns, (-2, -1))
            rotated_output = self.model(rotated_image, rotated_valid)
            equivariance = c4_equivariance_loss(output, rotated_output, quarter_turns)
            total = total + self.config.equivariance_weight * equivariance["equivariance_total"]
            metrics.update(equivariance)
        metrics["total"] = total
        return metrics

    def run_epoch(self, loader: Iterable[dict[str, Any]], training: bool) -> dict[str, float]:
        self.model.train(training)
        accumulated: dict[str, float] = {}
        batches = 0
        for batch in loader:
            batch = self._move(batch)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            context = torch.enable_grad() if training else torch.inference_mode()
            with context, torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                metrics = self._loss(batch, training)
            if training:
                self.scaler.scale(metrics["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            for key, value in metrics.items():
                accumulated[key] = accumulated.get(key, 0.0) + float(value.detach())
            batches += 1
        if batches == 0:
            raise ValueError("data loader produced no batches")
        return {key: value / batches for key, value in accumulated.items()}

    def fit(
        self,
        train_loader: Iterable[dict[str, Any]],
        validation_loader: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        log_path = self.output_directory / "metrics.jsonl"
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            train_metrics = self.run_epoch(train_loader, True)
            validation_metrics = (
                self.run_epoch(validation_loader, False) if validation_loader is not None else {}
            )
            self.scheduler.step()
            record = {
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation_metrics,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            validation_total = validation_metrics.get("total", train_metrics["total"])
            if validation_total < self.best_validation:
                self.best_validation = validation_total
                self.save(self.output_directory / "best.pt")
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save(self.output_directory / f"epoch_{epoch + 1:04d}.pt")
        self.save(self.output_directory / "last.pt")

    def save(self, path: str | Path) -> None:
        if self.run_config is None:
            raise ValueError("run_config is required for a contract-bearing checkpoint")
        torch.save(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "coordinate_contract": COORDINATE_CONTRACT,
                "model_contract": resolved_model_contract(self.model),
                "preprocessing_contract": resolved_preprocessing_contract(self.run_config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.epoch,
                "best_validation": self.best_validation,
                "trainer_config": asdict(self.config),
                "run_config": self.run_config,
            },
            path,
        )

    def resume(self, path: str | Path, *, allow_legacy: bool = False) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if self.run_config is None and not allow_legacy:
            raise ValueError("run_config is required to validate a resume checkpoint")
        if self.run_config is not None:
            validate_checkpoint_contract(
                checkpoint,
                model=self.model,
                run_config=self.run_config,
                switch_radius=self.config.switch_radius,
                allow_legacy=allow_legacy,
            )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("resume requires a full trainer checkpoint, not a raw state_dict")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = int(checkpoint["epoch"]) + 1
        self.best_validation = float(checkpoint["best_validation"])
