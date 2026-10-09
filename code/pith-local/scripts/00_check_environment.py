#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from racpith.provenance import atomic_write_json, runtime_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the registered Python 3.11 runtime")
    parser.add_argument("--require-conda-env", default="py311")
    parser.add_argument("--output")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"RAC-Pith requires Python 3.11, got {sys.version.split()[0]}")
    current = os.environ.get("CONDA_DEFAULT_ENV")
    if args.require_conda_env and current != args.require_conda_env:
        raise RuntimeError(
            f"expected conda environment {args.require_conda_env!r}, got {current!r}"
        )
    record = {
        "schema_version": "racpith.environment.v1",
        "status": "PASS",
        "geometry_device": "cpu",
        "geometry_dtype": "float64",
        "runtime": runtime_provenance(),
    }
    required = record["runtime"]["package_versions"]
    missing = [
        name
        for name in (
            "numpy",
            "scipy",
            "opencv-python",
            "pandas",
            "matplotlib",
            "scikit-learn",
            "joblib",
        )
        if required[name] is None
    ]
    if missing:
        raise RuntimeError(f"required distributions are missing: {missing}")
    if args.output:
        atomic_write_json(Path(args.output), record)
    print("environment PASS: Python 3.11, float64 CPU geometry dependencies available")


if __name__ == "__main__":
    main()
