from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from oapith.config import load_config
from oapith.contracts import DEFAULT_IMAGE_SIZE
from oapith.data import PithDataset, RandomSimilarity, collate_samples
from oapith.engine import Trainer, TrainerConfig
from oapith.models import OAPithNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the 2D-OA-Pith dense/probabilistic network")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Resume an old full checkpoint after manually verifying its semantics",
    )
    parser.add_argument("--device")
    return parser


def run_training(
    config_path: str | Path,
    *,
    resume: str | Path | None = None,
    allow_legacy_checkpoint: bool = False,
    device: str | torch.device | None = None,
) -> None:
    """Run training from Python without going through argument parsing."""
    config = load_config(config_path)
    data_config = config["data"]
    augmentation = RandomSimilarity(**data_config.get("augmentation", {}))
    shared = {
        "manifest": data_config["manifest"],
        "root": data_config.get("root", "."),
        "image_size": data_config.get("image_size", DEFAULT_IMAGE_SIZE),
        "mean": tuple(data_config.get("mean", [0.5, 0.5, 0.5])),
        "std": tuple(data_config.get("std", [0.25, 0.25, 0.25])),
        "raster_truncation": data_config.get("raster_truncation", 0.04),
        "state_switch_radius": config.get("training", {}).get("switch_radius", 2.0),
        "native_tile_probability": data_config.get("native_tile_probability", 0.5),
        "native_scale_jitter": tuple(
            data_config.get("native_scale_jitter", [0.8, 1.25])
        ),
    }
    train_dataset = PithDataset(
        **shared, split=data_config.get("train_split", "train"), augment=augmentation
    )
    validation_dataset = PithDataset(
        **shared, split=data_config.get("validation_split", "validation"), augment=None
    )
    loader_options = {
        "batch_size": data_config.get("batch_size", 4),
        "num_workers": data_config.get("workers", 4),
        # pin_memory 只为加速 CPU→GPU 拷贝；本管线 batch 很小（4×384²），收益
        # 可忽略，且 pin 线程在部分环境下崩溃。默认关闭，避免该问题。
        "pin_memory": bool(data_config.get("pin_memory", False)),
        "collate_fn": collate_samples,
        "persistent_workers": data_config.get("workers", 4) > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    model = OAPithNet(**config.get("model", {}))
    trainer = Trainer(
        model,
        TrainerConfig(**config.get("training", {})),
        Path(config.get("output_directory", "runs/oa_pith")),
        device=device,
        run_config=config,
    )
    if resume:
        trainer.resume(resume, allow_legacy=allow_legacy_checkpoint)
    trainer.fit(train_loader, validation_loader)


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        args.config,
        resume=args.resume,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
