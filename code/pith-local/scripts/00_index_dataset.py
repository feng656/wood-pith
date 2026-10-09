#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from racpith.config import load_config, resolve_runtime_paths, stable_hash
from racpith.data.urudendro4 import load_urudendro4
from racpith.provenance import atomic_write_json, atomic_write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly index the unmodified UruDendro4 v1 dataset")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "racpith_v1.json"),
        help="RAC-Pith config containing paths.dataset_root",
    )
    parser.add_argument(
        "--dataset-root",
        help="explicit override; defaults to paths.dataset_root in --config",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--pith-order", choices=["auto", "xy", "yx"], default="auto")
    parser.add_argument("--allow-nonofficial-subset", action="store_true")
    args = parser.parse_args()
    frozen = load_config(args.config)
    configured = resolve_runtime_paths(
        frozen,
        project_root=Path(__file__).resolve().parents[1],
    )
    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else configured.dataset_root
    )
    index = load_urudendro4(
        dataset_root,
        pith_order=args.pith_order,
        require_official_complete=not args.allow_nonofficial_subset,
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output / "source_manifest.jsonl", index.manifest_rows())
    audit = index.audit_summary()
    audit.update(
        {
            "runtime_config_path": str(frozen.source),
            "runtime_config_hash": frozen.sha256,
            "paths_config_hash": stable_hash(dict(frozen.section("paths"))),
            "configured_dataset_root": str(configured.dataset_root),
            "dataset_root_override_used": dataset_root != configured.dataset_root,
            "configured_output_root": str(configured.output_root),
            "configured_prepared_root": str(configured.prepared_root),
            "prepared_root_override_used": output.parent != configured.prepared_root,
        }
    )
    atomic_write_json(output / "dataset_audit.json", audit)
    print(
        f"indexed {len(index.samples)} sections / {len(index.tree_ids)} trees; "
        f"pith dialect={index.pith_dialect}, order={index.pith_order}"
    )


if __name__ == "__main__":
    main()
