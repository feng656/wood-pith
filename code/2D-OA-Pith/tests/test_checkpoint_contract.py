import copy

import pytest

from oapith import CHECKPOINT_SCHEMA_VERSION, COORDINATE_CONTRACT
from oapith.contracts import (
    resolved_model_contract,
    resolved_preprocessing_contract,
    validate_checkpoint_contract,
)
from oapith.models import OAPithNet


def _checkpoint(model: OAPithNet, config: dict[str, object]) -> dict[str, object]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "coordinate_contract": COORDINATE_CONTRACT,
        "model_contract": resolved_model_contract(model),
        "preprocessing_contract": resolved_preprocessing_contract(config),
        "trainer_config": {"switch_radius": 2.0},
        "run_config": copy.deepcopy(config),
    }


def test_resolved_contract_treats_explicit_defaults_as_equivalent() -> None:
    implicit = {"data": {}}
    explicit = {
        "data": {
            "image_size": 512,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
            "raster_truncation": 0.04,
            "native_tile_probability": 0.5,
            "native_scale_jitter": [0.8, 1.25],
            "augmentation": {
                "max_rotation_radians": 3.141592653589793,
                "min_scale": 0.85,
                "max_scale": 1.15,
                "max_translation": 0.10,
            },
        }
    }
    assert resolved_preprocessing_contract(implicit) == resolved_preprocessing_contract(explicit)


def test_current_schema_requires_complete_resolved_contract() -> None:
    model = OAPithNet(base_channels=4)
    config = {"data": {"image_size": 32}}
    checkpoint = _checkpoint(model, config)
    validate_checkpoint_contract(
        checkpoint, model=model, run_config=config, switch_radius=2.0
    )

    del checkpoint["preprocessing_contract"]
    with pytest.raises(ValueError, match="preprocessing"):
        validate_checkpoint_contract(
            checkpoint, model=model, run_config=config, switch_radius=2.0
        )


def test_model_contract_includes_base_width_and_nonshape_semantics() -> None:
    first = OAPithNet(base_channels=4, near_extent=2.5)
    second = OAPithNet(base_channels=8, near_extent=2.5)
    third = OAPithNet(base_channels=4, near_extent=3.0)
    assert resolved_model_contract(first) != resolved_model_contract(second)
    assert resolved_model_contract(first) != resolved_model_contract(third)
