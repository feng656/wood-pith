from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def _validate_finite_json(value: Any, *, location: str = "$") -> None:
    """Reject non-finite numbers, including finite-looking overflow literals.

    ``parse_constant`` catches the non-standard tokens ``NaN`` and
    ``Infinity``.  CPython's JSON decoder can additionally turn a valid JSON
    exponent such as ``1e999`` into ``inf``; walk the decoded value so that
    case is rejected as well.
    """

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, location=f"{location}.{key}")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    """Return whether *value* is a canonical lower-case SHA-256 digest."""

    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_source_tree(project_root: str | Path) -> str:
    """Hash the executable RAC-Pith source bundle independent of its location."""

    root = Path(project_root).expanduser().resolve()
    files = sorted(
        [path for path in (root / "src" / "racpith").rglob("*.py") if path.is_file()]
        + [path for path in (root / "scripts").rglob("*.py") if path.is_file()]
        + [path for path in (root / "scripts" / "manual").rglob("*.sh") if path.is_file()]
        + ([root / "pyproject.toml"] if (root / "pyproject.toml").is_file() else [])
    )
    if not files:
        raise ValueError(f"no RAC-Pith source files found under {root}")
    digest = hashlib.sha256()
    for path in files:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or path.is_symlink():
            raise ValueError(f"source bundle contains an unsafe path: {path}")
        relative = resolved.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, value: str) -> None:
    """Atomically replace a UTF-8 text artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json_object(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """Read a strict finite JSON object with path-aware errors."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    source = Path(path)
    try:
        with source.open("r", encoding=encoding) as handle:
            value = json.load(handle, parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {source}")
    try:
        _validate_finite_json(value)
    except ValueError as exc:
        raise ValueError(f"invalid strict JSON in {source}: {exc}") from exc
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid strict JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            try:
                _validate_finite_json(value)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid strict JSON: {exc}"
                ) from exc
            rows.append(value)
    return rows


def runtime_provenance() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "scipy",
        "opencv-python",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "joblib",
        "torch",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "package_versions": packages,
    }
