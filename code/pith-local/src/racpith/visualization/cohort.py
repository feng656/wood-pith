from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_cohort_figures(
    metrics: pd.DataFrame,
    output_dir: str | Path,
    risk_coverage: pd.DataFrame | None = None,
) -> list[Path]:
    output = Path(output_dir)
    paths: list[Path] = []

    point = metrics[(metrics["state"] == "POINT") & metrics["point_error_norm"].notna()].copy()
    figure, axis = plt.subplots(figsize=(7, 5))
    for stratum, frame in point.groupby("distance_stratum", dropna=False):
        values = np.sort(frame["point_error_norm"].to_numpy(dtype=float))
        if len(values):
            axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=str(stratum))
    axis.set_xlabel("POINT error / D_FOV")
    axis.set_ylabel("crop-level ECDF (descriptive only)")
    axis.legend()
    axis.grid(alpha=0.2)
    path = output / "point_error_ecdf.png"
    _save(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 5))
    order = ["POINT", "RANGE", "RAY", "AXIS", "MULTIMODAL", "REJECT"]
    table = pd.crosstab(metrics["distance_stratum"], metrics["state"], normalize="index").reindex(
        columns=order, fill_value=0.0
    )
    table.plot.bar(stacked=True, ax=axis, colormap="tab20c")
    axis.set_ylabel("fraction")
    axis.set_title("Geometry state by pith-to-crop distance")
    axis.legend(fontsize=7)
    path = output / "state_by_distance.png"
    _save(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 5))
    visible = metrics[
        metrics["condition_ratio"].notna()
        & (metrics["condition_ratio"].astype(float) > 0.0)
    ]
    for state, frame in visible.groupby("state"):
        axis.scatter(
            frame["condition_ratio"],
            frame["pith_outside_distance_norm"],
            s=10,
            alpha=0.45,
            label=state,
        )
    axis.set_xscale("log")
    axis.set_xlabel("H0 weak/strong eigenvalue ratio")
    axis.set_ylabel("pith outside distance / D_FOV")
    axis.legend(fontsize=7)
    path = output / "condition_distance_state.png"
    _save(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 5))
    risk = risk_coverage if risk_coverage is not None else pd.DataFrame()
    if not risk.empty:
        axis.plot(
            risk["point_coverage"],
            risk["false_point_risk"],
            color="#D55E00",
            lw=1.8,
        )
    else:
        axis.text(0.5, 0.5, "No ranked POINT operating points", ha="center", va="center")
    axis.set_xlabel("tree-equal POINT coverage")
    axis.set_ylabel("tree-equal False-POINT risk")
    axis.set_title("False-POINT risk–coverage")
    axis.grid(alpha=0.2)
    path = output / "risk_coverage.png"
    _save(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 5))
    support = metrics[metrics["support_contains_gt"].notna()].copy()
    if not support.empty:
        summary = support.groupby("state")["support_contains_gt"].mean().reindex(
            ["POINT", "RANGE", "RAY", "AXIS"], fill_value=0.0
        )
        summary.plot.bar(ax=axis, color="#56B4E9")
    else:
        axis.text(0.5, 0.5, "No evaluable support regions", ha="center", va="center")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("GT in profiled-objective support (crop descriptive)")
    axis.set_title("Support containment by structured state")
    path = output / "support_containment_by_state.png"
    _save(figure, path)
    paths.append(path)
    return paths
