from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .urudendro4 import UruDendro4Sample, parse_sample_code


@dataclass(frozen=True)
class TreeGroup:
    tree_id: str
    treatment: str
    block: str
    section_ids: tuple[str, ...]

    @property
    def section_count(self) -> int:
        return len(self.section_ids)


@dataclass(frozen=True)
class TreeSplitAssignment:
    tree_id: str
    split: str
    treatment: str
    block: str
    section_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "split": self.split,
            "treatment": self.treatment,
            "block": self.block,
            "section_ids": list(self.section_ids),
            "section_count": len(self.section_ids),
        }


@dataclass(frozen=True)
class SplitPlan:
    assignments: tuple[TreeSplitAssignment, ...]
    requested_fractions: Mapping[str, float]
    target_tree_counts: Mapping[str, int]
    objective: float
    random_seed: int
    search_trials: int
    manifest_sha256: str
    audit: Mapping[str, Any]

    @property
    def split_by_tree(self) -> dict[str, str]:
        return {assignment.tree_id: assignment.split for assignment in self.assignments}

    def tree_manifest_rows(self) -> list[dict[str, Any]]:
        return [assignment.as_dict() for assignment in self.assignments]

    def section_manifest_rows(
        self, samples: Sequence[UruDendro4Sample | Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        split_by_tree = self.split_by_tree
        result: list[dict[str, Any]] = []
        for row in _normalise_sample_rows(samples):
            enriched = dict(row)
            enriched["split"] = split_by_tree[row["tree_id"]]
            result.append(enriched)
        result.sort(key=lambda value: value["section_id"])
        return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_sample_rows(
    samples: Sequence[UruDendro4Sample | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not samples:
        raise ValueError("cannot split an empty sample collection")
    rows: list[dict[str, Any]] = []
    seen_sections: set[str] = set()
    for sample in samples:
        row = sample.to_manifest_row() if isinstance(sample, UruDendro4Sample) else dict(sample)
        if "section_id" not in row:
            raise ValueError("source manifest row is missing section_id")
        code = parse_sample_code(str(row["section_id"]))
        expected = code.as_dict()
        for key in ("tree_id", "treatment", "block", "tree_number", "height"):
            if key in row and str(row[key]) != expected[key]:
                raise ValueError(
                    f"section {code.section_id}: {key}={row[key]!r} conflicts with sample stem"
                )
            row[key] = expected[key]
        row["section_id"] = code.section_id
        if code.section_id in seen_sections:
            raise ValueError(f"duplicate source section {code.section_id!r}")
        seen_sections.add(code.section_id)
        rows.append(row)
    rows.sort(key=lambda value: value["section_id"])
    return rows


def build_tree_groups(
    samples: Sequence[UruDendro4Sample | Mapping[str, Any]],
) -> tuple[TreeGroup, ...]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _normalise_sample_rows(samples):
        grouped.setdefault(row["tree_id"], []).append(row)
    result: list[TreeGroup] = []
    for tree_id, rows in sorted(grouped.items()):
        treatments = {str(row["treatment"]) for row in rows}
        blocks = {str(row["block"]) for row in rows}
        if len(treatments) != 1 or len(blocks) != 1:
            raise ValueError(f"tree {tree_id} has inconsistent treatment/block metadata")
        result.append(
            TreeGroup(
                tree_id=tree_id,
                treatment=next(iter(treatments)),
                block=next(iter(blocks)),
                section_ids=tuple(sorted(str(row["section_id"]) for row in rows)),
            )
        )
    return tuple(result)


def _target_counts(total: int, fractions: Mapping[str, float]) -> dict[str, int]:
    if len(fractions) < 2:
        raise ValueError("at least two splits are required")
    names = list(fractions)
    values = [float(fractions[name]) for name in names]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("split fractions must be finite and strictly positive")
    scale = sum(values)
    if not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"split fractions must sum to one, got {scale}")
    raw = [total * value / scale for value in values]
    counts = [int(math.floor(value)) for value in raw]
    remaining = total - sum(counts)
    priority = sorted(range(len(names)), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in priority[:remaining]:
        counts[index] += 1
    if any(count <= 0 for count in counts):
        raise ValueError(
            f"too few biological groups for requested split fractions: {dict(zip(names, counts))}"
        )
    return dict(zip(names, counts, strict=True))


def _candidate_order(groups: Sequence[TreeGroup], seed: int, trial: int) -> list[TreeGroup]:
    def key(group: TreeGroup) -> tuple[bytes, str]:
        digest = hashlib.sha256(f"{seed}|{trial}|{group.tree_id}".encode("utf-8")).digest()
        return digest, group.tree_id

    return sorted(groups, key=key)


def _assignment_from_order(
    ordered: Sequence[TreeGroup], split_names: Sequence[str], counts: Mapping[str, int]
) -> dict[str, str]:
    result: dict[str, str] = {}
    cursor = 0
    for split in split_names:
        stop = cursor + counts[split]
        for group in ordered[cursor:stop]:
            result[group.tree_id] = split
        cursor = stop
    if cursor != len(ordered):
        raise AssertionError("internal split count mismatch")
    return result


def _balance_objective(
    groups: Sequence[TreeGroup],
    assignment: Mapping[str, str],
    fractions: Mapping[str, float],
    counts: Mapping[str, int],
    stratify_keys: Sequence[str],
) -> float:
    split_names = list(fractions)
    total_groups = len(groups)
    total_sections = sum(group.section_count for group in groups)
    objective = 0.0

    for field in stratify_keys:
        if field not in {"treatment", "block"}:
            raise ValueError(f"unsupported stratification key: {field!r}")
        categories = sorted({getattr(group, field) for group in groups})
        totals = {
            category: sum(getattr(group, field) == category for group in groups)
            for category in categories
        }
        for split in split_names:
            for category in categories:
                observed = sum(
                    assignment[group.tree_id] == split and getattr(group, field) == category
                    for group in groups
                )
                expected = totals[category] * counts[split] / total_groups
                objective += (observed - expected) ** 2 / max(expected, 0.25)
            if counts[split] >= len(categories):
                missing = sum(
                    not any(
                        assignment[group.tree_id] == split
                        and getattr(group, field) == category
                        for group in groups
                    )
                    for category in categories
                )
                objective += 1000.0 * missing

    if len(stratify_keys) > 1:
        cells = sorted({tuple(getattr(group, key) for key in stratify_keys) for group in groups})
        totals = {
            cell: sum(tuple(getattr(group, key) for key in stratify_keys) == cell for group in groups)
            for cell in cells
        }
        for split in split_names:
            for cell in cells:
                observed = sum(
                    assignment[group.tree_id] == split
                    and tuple(getattr(group, key) for key in stratify_keys) == cell
                    for group in groups
                )
                expected = totals[cell] * counts[split] / total_groups
                objective += 0.5 * (observed - expected) ** 2 / max(expected, 0.25)

    for split in split_names:
        observed_sections = sum(
            group.section_count for group in groups if assignment[group.tree_id] == split
        )
        expected_sections = total_sections * float(fractions[split])
        objective += 0.25 * (observed_sections - expected_sections) ** 2 / max(
            expected_sections, 1.0
        )
    return float(objective)


def _assignment_signature(assignment: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((tree_id, split) for tree_id, split in assignment.items()))


def audit_split_assignments(
    samples: Sequence[UruDendro4Sample | Mapping[str, Any]],
    assignments: Sequence[TreeSplitAssignment],
) -> dict[str, Any]:
    rows = _normalise_sample_rows(samples)
    expected_groups = {group.tree_id: group for group in build_tree_groups(samples)}
    expected_trees = set(expected_groups)
    assignment_by_tree: dict[str, str] = {}
    for assignment in assignments:
        if assignment.tree_id in assignment_by_tree:
            raise ValueError(f"tree {assignment.tree_id} has more than one assignment row")
        assignment_by_tree[assignment.tree_id] = assignment.split
        expected = expected_groups.get(assignment.tree_id)
        if expected is None:
            continue
        if (
            assignment.treatment != expected.treatment
            or assignment.block != expected.block
            or tuple(assignment.section_ids) != expected.section_ids
        ):
            raise ValueError(
                f"tree assignment metadata for {assignment.tree_id} disagrees with source sections"
            )
    if set(assignment_by_tree) != expected_trees:
        raise ValueError(
            "split assignments do not cover source trees exactly; "
            f"missing={sorted(expected_trees - set(assignment_by_tree))}, "
            f"extra={sorted(set(assignment_by_tree) - expected_trees)}"
        )

    split_to_trees: dict[str, set[str]] = {}
    split_to_sections: dict[str, set[str]] = {}
    treatment_counts: dict[str, dict[str, int]] = {}
    block_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        split = assignment_by_tree[row["tree_id"]]
        split_to_trees.setdefault(split, set()).add(row["tree_id"])
        split_to_sections.setdefault(split, set()).add(row["section_id"])
        treatment_counts.setdefault(split, {})[row["treatment"]] = (
            treatment_counts.setdefault(split, {}).get(row["treatment"], 0) + 1
        )
        block_counts.setdefault(split, {})[row["block"]] = (
            block_counts.setdefault(split, {}).get(row["block"], 0) + 1
        )

    names = sorted(split_to_trees)
    tree_intersections: dict[str, list[str]] = {}
    section_intersections: dict[str, list[str]] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pair = f"{left}|{right}"
            tree_intersections[pair] = sorted(split_to_trees[left] & split_to_trees[right])
            section_intersections[pair] = sorted(
                split_to_sections[left] & split_to_sections[right]
            )
    leakage_free = not any(tree_intersections.values()) and not any(
        section_intersections.values()
    )
    if not leakage_free:
        raise ValueError("tree or section leakage detected between splits")

    tree_rows = [assignment.as_dict() for assignment in sorted(assignments, key=lambda x: x.tree_id)]
    return {
        "schema_version": "racpith.split_audit.v1",
        "tree_count": len(expected_trees),
        "section_count": len(rows),
        "tree_counts": {name: len(split_to_trees[name]) for name in names},
        "section_counts": {name: len(split_to_sections[name]) for name in names},
        "section_counts_by_treatment": {
            name: dict(sorted(treatment_counts[name].items())) for name in names
        },
        "section_counts_by_block": {
            name: dict(sorted(block_counts[name].items())) for name in names
        },
        "tree_intersections": tree_intersections,
        "section_intersections": section_intersections,
        "leakage_free": leakage_free,
        "assignment_sha256": _canonical_sha256(tree_rows),
    }


def make_grouped_split(
    samples: Sequence[UruDendro4Sample | Mapping[str, Any]],
    split_fractions: Mapping[str, float],
    *,
    random_seed: int = 20260906,
    search_trials: int = 20_000,
    stratify_keys: Sequence[str] = ("treatment", "block"),
) -> SplitPlan:
    """Assign whole biological trees to deterministic, leakage-free splits."""

    if search_trials < 1:
        raise ValueError("search_trials must be at least one")
    groups = build_tree_groups(samples)
    counts = _target_counts(len(groups), split_fractions)
    split_names = list(split_fractions)

    best_assignment: dict[str, str] | None = None
    best_objective = math.inf
    best_signature: tuple[tuple[str, str], ...] | None = None
    for trial in range(search_trials):
        candidate = _assignment_from_order(
            _candidate_order(groups, random_seed, trial), split_names, counts
        )
        objective = _balance_objective(
            groups, candidate, split_fractions, counts, stratify_keys
        )
        signature = _assignment_signature(candidate)
        if (
            objective < best_objective - 1e-12
            or (
                abs(objective - best_objective) <= 1e-12
                and (best_signature is None or signature < best_signature)
            )
        ):
            best_assignment = candidate
            best_objective = objective
            best_signature = signature
    if best_assignment is None:
        raise AssertionError("deterministic split search produced no candidate")

    assignments = tuple(
        TreeSplitAssignment(
            tree_id=group.tree_id,
            split=best_assignment[group.tree_id],
            treatment=group.treatment,
            block=group.block,
            section_ids=group.section_ids,
        )
        for group in groups
    )
    audit = audit_split_assignments(samples, assignments)
    manifest_sha256 = str(audit["assignment_sha256"])
    return SplitPlan(
        assignments=assignments,
        requested_fractions={name: float(split_fractions[name]) for name in split_names},
        target_tree_counts=counts,
        objective=best_objective,
        random_seed=int(random_seed),
        search_trials=int(search_trials),
        manifest_sha256=manifest_sha256,
        audit={
            **audit,
            "requested_fractions": {
                name: float(split_fractions[name]) for name in split_names
            },
            "target_tree_counts": counts,
            "balance_objective": best_objective,
            "random_seed": int(random_seed),
            "search_trials": int(search_trials),
            "stratify_keys": list(stratify_keys),
        },
    )


def make_train_test_split(
    samples: Sequence[UruDendro4Sample | Mapping[str, Any]],
    *,
    train_fraction: float = 0.75,
    train_name: str = "train",
    test_name: str = "sealed_test",
    random_seed: int = 20260906,
    search_trials: int = 20_000,
) -> SplitPlan:
    return make_grouped_split(
        samples,
        {train_name: train_fraction, test_name: 1.0 - train_fraction},
        random_seed=random_seed,
        search_trials=search_trials,
    )


def make_development_calibration_sealed_split(
    samples: Sequence[UruDendro4Sample | Mapping[str, Any]],
    *,
    development_fraction: float = 2.0 / 3.0,
    calibration_fraction: float = 1.0 / 6.0,
    sealed_fraction: float = 1.0 / 6.0,
    sealed_name: str = "sealed_test",
    random_seed: int = 20260906,
    search_trials: int = 20_000,
) -> SplitPlan:
    return make_grouped_split(
        samples,
        {
            "development": development_fraction,
            "calibration": calibration_fraction,
            sealed_name: sealed_fraction,
        },
        random_seed=random_seed,
        search_trials=search_trials,
    )


def make_grouped_folds(
    samples: Sequence[UruDendro4Sample | Mapping[str, Any]],
    *,
    n_folds: int,
    random_seed: int = 20260906,
    search_trials: int = 20_000,
) -> SplitPlan:
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    fractions = {f"fold_{index:02d}": 1.0 / n_folds for index in range(n_folds)}
    return make_grouped_split(
        samples,
        fractions,
        random_seed=random_seed,
        search_trials=search_trials,
    )


def iter_section_assignments(
    plan: SplitPlan,
    samples: Sequence[UruDendro4Sample | Mapping[str, Any]],
) -> Iterable[dict[str, Any]]:
    yield from plan.section_manifest_rows(samples)
