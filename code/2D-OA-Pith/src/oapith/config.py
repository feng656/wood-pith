from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    training_switch = value.get("training", {}).get("switch_radius")
    geometry_switch = value.get("geometry", {}).get("switch_radius")
    if (
        training_switch is not None
        and geometry_switch is not None
        and float(training_switch) != float(geometry_switch)
    ):
        raise ValueError(
            "training.switch_radius and geometry.switch_radius must match: "
            "the network uses eta=switch_radius/d while geometry uses rho=1/d"
        )
    return value
