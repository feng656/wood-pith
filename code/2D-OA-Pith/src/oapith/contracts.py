from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from oapith import CHECKPOINT_SCHEMA_VERSION, COORDINATE_CONTRACT


DEFAULT_IMAGE_SIZE = 512


def resolved_model_contract(model: object) -> dict[str, object]:
    """Constructor semantics that are not reliably recoverable from tensor shapes."""
    return {
        "class": type(model).__name__,
        "base_channels": getattr(model, "base_channels", None),
        "mixture_components": getattr(model, "mixture_components", None),
        "near_extent": getattr(model, "near_extent", None),
        "min_scale": getattr(model, "min_scale", None),
        "detach_variance_features": getattr(model, "detach_variance_features", None),
        "embedding_dim": getattr(model, "embedding_dim", None),
        "defect_classes": getattr(model, "defect_classes", None),
    }


def resolved_preprocessing_contract(run_config: Mapping[str, Any]) -> dict[str, object]:
    """Resolve data defaults before serializing/comparing a checkpoint contract.

    Comparing raw YAML dictionaries is unsafe: an omitted key and an explicitly
    supplied default have identical runtime semantics, while different call-site
    defaults can make two omitted keys behave differently.
    """
    data = run_config.get("data", {})
    if not isinstance(data, Mapping):
        raise ValueError("run_config.data must be a mapping")
    augmentation = data.get("augmentation", {})
    if not isinstance(augmentation, Mapping):
        raise ValueError("run_config.data.augmentation must be a mapping")

    mean = tuple(float(value) for value in data.get("mean", [0.5, 0.5, 0.5]))
    std = tuple(float(value) for value in data.get("std", [0.25, 0.25, 0.25]))
    jitter = tuple(float(value) for value in data.get("native_scale_jitter", [0.8, 1.25]))
    if len(mean) != 3 or len(std) != 3 or len(jitter) != 2:
        raise ValueError("invalid normalization or native-scale contract dimensions")
    return {
        "image_size": int(data.get("image_size", DEFAULT_IMAGE_SIZE)),
        "mean": list(mean),
        "std": list(std),
        "raster_truncation": float(data.get("raster_truncation", 0.04)),
        "native_tile_probability": float(data.get("native_tile_probability", 0.5)),
        "native_scale_jitter": list(jitter),
        "augmentation": {
            "max_rotation_radians": float(
                augmentation.get("max_rotation_radians", 3.141592653589793)
            ),
            "min_scale": float(augmentation.get("min_scale", 0.85)),
            "max_scale": float(augmentation.get("max_scale", 1.15)),
            "max_translation": float(augmentation.get("max_translation", 0.10)),
        },
    }


def validate_checkpoint_contract(
    checkpoint: object,
    *,
    model: object,
    run_config: Mapping[str, Any],
    switch_radius: float,
    allow_legacy: bool = False,
) -> None:
    """Validate all coordinate/model/preprocessing semantics before state loading."""
    if not isinstance(checkpoint, Mapping):
        if allow_legacy:
            return
        raise ValueError("checkpoint is not a contract-bearing mapping")

    schema = checkpoint.get("checkpoint_schema_version")
    if schema != CHECKPOINT_SCHEMA_VERSION:
        if allow_legacy:
            return
        raise ValueError(
            "checkpoint schema is missing or incompatible; allow a legacy "
            "checkpoint only after manually verifying its semantics"
        )
    if checkpoint.get("coordinate_contract") != COORDINATE_CONTRACT:
        raise ValueError("checkpoint coordinate contract is incompatible")
    if checkpoint.get("model_contract") != resolved_model_contract(model):
        raise ValueError("checkpoint/config resolved model contract mismatch")
    if checkpoint.get("preprocessing_contract") != resolved_preprocessing_contract(run_config):
        raise ValueError("checkpoint/config resolved preprocessing contract mismatch")
    if not isinstance(checkpoint.get("run_config"), Mapping):
        raise ValueError("checkpoint lacks its training run_config provenance")
    trainer_config = checkpoint.get("trainer_config")
    if not isinstance(trainer_config, Mapping) or "switch_radius" not in trainer_config:
        raise ValueError("checkpoint lacks the training switch-radius contract")
    if float(trainer_config["switch_radius"]) != float(switch_radius):
        raise ValueError(
            "checkpoint/config switch_radius mismatch would rescale every far proposal"
        )
