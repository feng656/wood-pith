from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .provenance import read_json_object


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class FrozenConfig:
    """Immutable-by-convention configuration plus its content hash."""

    data: Mapping[str, Any]
    source: Path
    sha256: str

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, Mapping):
            raise KeyError(f"configuration section {name!r} is missing or not an object")
        return value


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved runtime paths owned by one RAC-Pith configuration."""

    dataset_root: Path
    output_root: Path
    prepared_root: Path
    development_root: Path
    sealed_root: Path


def _configured_path_text(section: Mapping[str, Any], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"configuration paths.{name} must be a non-empty string")
    if any(character in value for character in ("\n", "\r", "\0")):
        raise ValueError(f"configuration paths.{name} contains a control character")
    return value.strip()


def _safe_output_child(output_root: Path, section: Mapping[str, Any], name: str) -> Path:
    raw = Path(_configured_path_text(section, name))
    if raw.is_absolute() or ".." in raw.parts or len(raw.parts) != 1:
        raise ValueError(
            f"configuration paths.{name} must be one safe relative directory name"
        )
    resolved = (output_root / raw).resolve()
    if resolved == output_root or not resolved.is_relative_to(output_root):
        raise ValueError(f"configuration paths.{name} must be below output_root")
    return resolved


def resolve_runtime_paths(
    config: FrozenConfig,
    *,
    project_root: str | Path,
) -> RuntimePaths:
    """Resolve the single configured dataset/output layout.

    ``output_root`` is interpreted relative to the project root, never relative
    to whichever directory happened to launch a shell script.
    """

    section = config.section("paths")
    root = Path(project_root).expanduser().resolve()
    dataset_root = Path(_configured_path_text(section, "dataset_root")).expanduser()
    if not dataset_root.is_absolute():
        raise ValueError("configuration paths.dataset_root must be absolute")
    dataset_root = dataset_root.resolve()

    output_value = Path(_configured_path_text(section, "output_root")).expanduser()
    output_root = (
        output_value.resolve()
        if output_value.is_absolute()
        else (root / output_value).resolve()
    )
    if output_root == dataset_root or output_root.is_relative_to(dataset_root):
        raise ValueError("configuration paths.output_root must be outside dataset_root")
    prepared_root = _safe_output_child(
        output_root,
        section,
        "prepared_subdirectory",
    )
    development_root = _safe_output_child(
        output_root,
        section,
        "development_subdirectory",
    )
    sealed_root = _safe_output_child(
        output_root,
        section,
        "sealed_subdirectory",
    )
    if len({prepared_root, development_root, sealed_root}) != 3:
        raise ValueError("configured prepared/development/sealed directories must be distinct")
    return RuntimePaths(
        dataset_root=dataset_root,
        output_root=output_root,
        prepared_root=prepared_root,
        development_root=development_root,
        sealed_root=sealed_root,
    )


def load_config(path: str | Path) -> FrozenConfig:
    source = Path(path).expanduser().resolve()
    data = read_json_object(source)
    schema = data.get("schema_version")
    if schema != "racpith.config.v1":
        raise ValueError(f"unsupported RAC-Pith configuration schema: {schema!r}")
    payload = _canonical_json(data).encode("utf-8")
    return FrozenConfig(data=data, source=source, sha256=hashlib.sha256(payload).hexdigest())


def stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
