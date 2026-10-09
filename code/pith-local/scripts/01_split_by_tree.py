#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from racpith.data.split import make_grouped_folds, make_grouped_split
from racpith.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json_object,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split UruDendro4 before generating any crops")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_json_object(args.config)
    if config.get("schema_version") != "racpith.split.v1":
        raise ValueError("unsupported split configuration schema")
    source = read_jsonl(args.source_manifest)
    plan = make_grouped_split(
        source,
        config["splits"],
        random_seed=int(config["random_seed"]),
        search_trials=int(config["search_trials"]),
        stratify_keys=tuple(config.get("stratify_keys", ["treatment", "block"])),
    )
    sections = plan.section_manifest_rows(source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output / "tree_split.jsonl", plan.tree_manifest_rows())
    atomic_write_jsonl(output / "section_manifest.jsonl", sections)
    atomic_write_json(output / "split_summary.json", dict(plan.audit))

    folds = int(config.get("grouped_cv_folds_within_train", 0))
    if folds:
        train_names = [name for name in config["splits"] if name in {"train", "development"}]
        if len(train_names) != 1:
            raise ValueError("grouped_cv_folds_within_train requires exactly one train/development split")
        train = [row for row in sections if row["split"] == train_names[0]]
        fold_plan = make_grouped_folds(
            train,
            n_folds=folds,
            random_seed=int(config["random_seed"]),
            search_trials=int(config["search_trials"]),
        )
        fold_by_tree = fold_plan.split_by_tree
        sections = [
            {**row, "inner_fold": fold_by_tree.get(row["tree_id"])} for row in sections
        ]
        atomic_write_jsonl(output / "section_manifest.jsonl", sections)
        atomic_write_jsonl(output / "train_fold_assignments.jsonl", fold_plan.tree_manifest_rows())
        atomic_write_json(output / "train_fold_audit.json", dict(fold_plan.audit))
    print(
        f"split {len(plan.assignments)} trees; assignment_sha256={plan.manifest_sha256}; "
        f"leakage_free={plan.audit['leakage_free']}"
    )


if __name__ == "__main__":
    main()
