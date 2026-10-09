from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .contracts import EvidenceBundle


@dataclass(frozen=True)
class DeletionGroup:
    group_id: str
    level: str
    ring_id: str
    arc_id: str | None
    scale_fraction: float | None
    phase: float | None
    interval_fraction: tuple[float, float] | None
    remove_mask: np.ndarray
    removed_mass: float
    partition_id: str

    @property
    def mask_hash(self) -> str:
        packed = np.packbits(self.remove_mask.astype(np.uint8), bitorder="little")
        return hashlib.sha256(packed.tobytes()).hexdigest()


def _group_id(parts: Iterable[object]) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def build_deletion_groups(
    bundle: EvidenceBundle,
    contribution_cfg: Mapping[str, object],
    include_audit_partitions: bool = False,
) -> list[DeletionGroup]:
    groups: list[DeletionGroup] = []
    levels = set(str(value) for value in contribution_cfg["levels"])
    if "ring" in levels:
        for ring_index, ring_id in enumerate(bundle.ring_ids):
            mask = bundle.ring_index == ring_index
            groups.append(
                DeletionGroup(
                    group_id=_group_id((bundle.crop_id, "ring", ring_id)),
                    level="ring",
                    ring_id=ring_id,
                    arc_id=None,
                    scale_fraction=None,
                    phase=None,
                    interval_fraction=None,
                    remove_mask=mask,
                    removed_mass=float(bundle.base_weight[mask].sum()),
                    partition_id="ring",
                )
            )
    if "arc" in levels:
        for arc_index, arc_id in enumerate(bundle.arc_ids):
            mask = bundle.arc_index == arc_index
            ring_values = np.unique(bundle.ring_index[mask])
            if len(ring_values) != 1:
                raise ValueError(f"arc {arc_id} maps to more than one parent ring")
            ring_id = bundle.ring_ids[int(ring_values[0])]
            groups.append(
                DeletionGroup(
                    group_id=_group_id((bundle.crop_id, "arc", arc_id)),
                    level="arc",
                    ring_id=ring_id,
                    arc_id=arc_id,
                    scale_fraction=None,
                    phase=None,
                    interval_fraction=None,
                    remove_mask=mask,
                    removed_mass=float(bundle.base_weight[mask].sum()),
                    partition_id="arc",
                )
            )
    if "subarc" not in levels:
        return groups

    primary = float(contribution_cfg["primary_subarc_fraction"])
    fractions = [primary]
    phases = [0.0]
    if include_audit_partitions:
        fractions = sorted(
            {
                primary,
                *(float(value) for value in contribution_cfg["audit_subarc_fractions"]),
            }
        )
        phases = sorted(
            {0.0, *(float(value) for value in contribution_cfg["audit_phases"])}
        )
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"subarc fraction must lie in (0,1], got {fraction}")
        for phase in phases:
            if not 0 <= phase < 1:
                raise ValueError(f"subarc phase must lie in [0,1), got {phase}")
            partition_id = f"subarc:f={fraction:.8g}:phase={phase:.8g}"
            for arc_index, arc_id in enumerate(bundle.arc_ids):
                arc_mask = bundle.arc_index == arc_index
                ring_values = np.unique(bundle.ring_index[arc_mask])
                if len(ring_values) != 1:
                    raise ValueError(f"arc {arc_id} maps to more than one parent ring")
                ring_id = bundle.ring_ids[int(ring_values[0])]
                values = np.minimum(bundle.arc_fraction[arc_mask], np.nextafter(1.0, 0.0))
                bin_index = np.floor((values - phase * fraction) / fraction).astype(np.int64)
                for index in np.unique(bin_index):
                    lo = max(0.0, (float(index) + phase) * fraction)
                    hi = min(1.0, (float(index) + phase + 1.0) * fraction)
                    if hi <= lo:
                        continue
                    local = np.zeros(bundle.n_nodes, dtype=bool)
                    arc_positions = np.flatnonzero(arc_mask)
                    local[arc_positions[bin_index == index]] = True
                    if not np.any(local):
                        continue
                    groups.append(
                        DeletionGroup(
                            group_id=_group_id(
                                (bundle.crop_id, "subarc", arc_id, fraction, phase, int(index))
                            ),
                            level="subarc",
                            ring_id=ring_id,
                            arc_id=arc_id,
                            scale_fraction=fraction,
                            phase=phase,
                            interval_fraction=(lo, hi),
                            remove_mask=local,
                            removed_mass=float(bundle.base_weight[local].sum()),
                            partition_id=partition_id,
                        )
                    )
    return groups


def validate_partition_mass(bundle: EvidenceBundle, groups: list[DeletionGroup]) -> list[str]:
    """Return human-readable failures; overlapping levels are checked separately."""
    failures: list[str] = []
    by_partition: dict[str, list[DeletionGroup]] = {}
    for group in groups:
        by_partition.setdefault(group.partition_id, []).append(group)
    total_mass = float(bundle.base_weight.sum())
    for partition_id, partition in by_partition.items():
        cover = np.zeros(bundle.n_nodes, dtype=np.int64)
        for group in partition:
            cover += group.remove_mask.astype(np.int64)
        if np.any(cover != 1):
            failures.append(f"{partition_id}: deletion groups do not form a one-cover partition")
        mass = sum(group.removed_mass for group in partition)
        if abs(mass - total_mass) > max(1e-10, total_mass * 1e-8):
            failures.append(f"{partition_id}: group mass {mass} differs from evidence mass {total_mass}")
    return failures
