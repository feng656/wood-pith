from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ERROR_NORM_BINS = (
    ("[0, 0.01)", 0.0, 0.01),
    ("[0.01, 0.03)", 0.01, 0.03),
    ("[0.03, 0.10)", 0.03, 0.10),
    ("[0.10, 0.25)", 0.10, 0.25),
    ("[0.25, +inf)", 0.25, math.inf),
)

CONTRIBUTION_COLORS = {
    "BENEFICIAL_GT": "#009E73",
    "HARMFUL_GT": "#D55E00",
    "POSITIVE_RAW": "#009E73",
    "NEGATIVE_RAW": "#D55E00",
    "NEUTRAL": "#7A7A7A",
    "UNCERTAIN": "#E6A700",
}

# Geometric pointing-agreement display (diagnostic only, see agreement_value):
# mean |cos| between each subarc's sampled normals and the direction to the
# estimated (consensus) pith. 1.0 = the segment points exactly at the position
# the other rings point to; values far below 1.0 deviate from that consensus.
AGREEMENT_HIGH = 0.90
AGREEMENT_MID = 0.70
AGREEMENT_LOW = 0.40
AGREEMENT_COUNTS = ("AGREE_HIGH", "AGREE_MID", "DEVIATE", "DEVIATE_STRONG", "NO_DATA")

# Sub-arc explanation categories from scripts/22_explain_subarcs.py
EXPLAIN_COLORS = {
    "SUPPORTING": "#009E73",
    "SUPPORTING_SHORT": "#88CCAA",
    "MODERATE": "#E6A700",
    "BOUNDARY_TRUNCATED": "#7A7A7A",
    "DEVIATING": "#D55E00",
    "NO_CENTER": "#4C78A8",
}
EXPLAIN_LABELS = {
    "SUPPORTING": "指向一致+有效弧长(支撑共识)",
    "SUPPORTING_SHORT": "指向一致但弧长不足",
    "MODERATE": "指向中间",
    "BOUNDARY_TRUNCATED": "贴边截断(约束弱)",
    "DEVIATING": "指向偏离(冲突)",
    "NO_CENTER": "无最终坐标",
}
EXPLAIN_LABELS_EN = {
    "SUPPORTING": "aligned + long (supports)",
    "SUPPORTING_SHORT": "aligned but short",
    "MODERATE": "moderate pointing",
    "BOUNDARY_TRUNCATED": "boundary-truncated (weak)",
    "DEVIATING": "deviating (conflict)",
    "NO_CENTER": "no final coordinate",
}

_CJK_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
)


def setup_cjk_font() -> bool:
    """Register a CJK font for matplotlib (Windows fonts or RACPITH_CJK_FONT); return success."""
    import os

    from matplotlib import font_manager

    candidates = list(_CJK_CANDIDATES)
    env_font = os.environ.get("RACPITH_CJK_FONT")
    if env_font:
        candidates.insert(0, env_font)
    for path in candidates:
        if Path(path).is_file():
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                continue
    return False


def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.asarray([int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _rgb_to_hex(rgb: np.ndarray) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02X}" for c in rgb)


def blend(color_a: str, color_b: str, t: float) -> str:
    a = _hex_to_rgb(color_a)
    b = _hex_to_rgb(color_b)
    return _rgb_to_hex(a + (b - a) * float(t))


def agreement_color(value: float) -> str:
    """Continuous green -> yellow -> orange for subarc pointing agreement."""
    if value >= AGREEMENT_HIGH:
        return "#009E73"
    if value >= AGREEMENT_MID:
        return blend("#E6A700", "#009E73", (value - AGREEMENT_MID) / (AGREEMENT_HIGH - AGREEMENT_MID))
    if value >= AGREEMENT_LOW:
        return blend("#D55E00", "#E6A700", (value - AGREEMENT_LOW) / (AGREEMENT_MID - AGREEMENT_LOW))
    return "#D55E00"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def outside_marker(ax, point, width, height, color, label):
    x, y = map(float, point)
    if 0 <= x < width and 0 <= y < height:
        ax.scatter([x], [y], s=70, marker="x", linewidths=2, color=color, label=label)
        return
    center = np.array([width / 2.0, height / 2.0])
    vector = np.array([x, y]) - center
    scale = min(0.44 * width / max(abs(vector[0]), 1e-9), 0.44 * height / max(abs(vector[1]), 1e-9))
    end = center + max(min(scale, 1.0), 0.0) * vector
    ax.annotate(f"{label} outside ({np.linalg.norm(vector):.0f}px)", xy=end, xytext=center,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 2}, color=color, fontsize=8)


def arc_segment(points: np.ndarray, interval: list[float] | None) -> np.ndarray:
    if interval is None or len(points) < 2:
        return points
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    length = np.r_[0.0, np.cumsum(step)]
    if length[-1] <= 0.0:
        return points[:1]
    fraction = length / length[-1]
    lo = float(np.clip(interval[0], 0.0, 1.0))
    hi = float(np.clip(interval[1], 0.0, 1.0))
    interior = fraction[(fraction > lo) & (fraction < hi)]
    sample = np.unique(np.r_[lo, interior, hi])
    return np.column_stack([
        np.interp(sample, fraction, points[:, 0]),
        np.interp(sample, fraction, points[:, 1]),
    ])


def evidence_mask(bundle: dict, arc_id: str, interval: list[float] | None) -> np.ndarray | None:
    """Boolean mask over evidence rows for one contribution group."""
    arc_index = bundle["arc_index"].get(str(arc_id))
    if arc_index is None:
        return None
    mask = bundle["arc_indices"] == int(arc_index)
    if interval is not None:
        lo = float(np.clip(interval[0], 0.0, 1.0))
        hi = float(np.clip(interval[1], 0.0, 1.0))
        mask &= bundle["arc_fractions"] >= lo - 1e-12
        mask &= bundle["arc_fractions"] <= hi + 1e-12
    return mask


def evidence_segment(bundle: dict, arc_id: str, interval: list[float] | None) -> np.ndarray | None:
    """Return the exact sampled nodes used to compute a contribution group."""
    mask = evidence_mask(bundle, arc_id, interval)
    if mask is None:
        return None
    points = bundle["points_crop_px"][mask]
    return points if len(points) >= 2 else None


def agreement_value(
    bundle: dict,
    arc_id: str,
    interval: list[float] | None,
    center: np.ndarray | list[float] | None,
) -> float | None:
    """Mean |cos| between the group's sampled normals and the direction to `center`.

    This implements the user's "does this small segment point to the same
    position as the other rings" criterion: for every sampled node of the
    subarc, the ring normal (perpendicular of the spline tangent) is compared
    with the direction toward the consensus pith. Returns None when the group
    has fewer than two sampled nodes or no center is available.
    """
    if center is None:
        return None
    mask = evidence_mask(bundle, arc_id, interval)
    if mask is None or mask.sum() < 2:
        return None
    direction = np.asarray(center, dtype=float) - bundle["points_crop_px"][mask]
    lengths = np.linalg.norm(direction, axis=1, keepdims=True)
    lengths[lengths < 1e-9] = 1e-9
    unit = direction / lengths
    tangents = bundle["tangents"][mask]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return float(np.mean(np.abs(np.sum(normals * unit, axis=1))))


def contribution_label(record: dict, allow_point_estimate: bool, raw_sign: bool) -> str:
    formal = str(record.get("gt_label") or "UNCERTAIN")
    if formal in {"BENEFICIAL_GT", "HARMFUL_GT", "NEUTRAL"}:
        return formal
    if allow_point_estimate:
        point = str(record.get("gt_point_label") or "UNCERTAIN")
        if point in {"BENEFICIAL_GT", "HARMFUL_GT", "NEUTRAL"}:
            if raw_sign and point == "NEUTRAL" and record.get("contrib_gt_norm") is not None:
                value = float(record["contrib_gt_norm"])
                if value > 0.0:
                    return "POSITIVE_RAW"
                if value < 0.0:
                    return "NEGATIVE_RAW"
            return point
    return "UNCERTAIN"


def render_contribution_card(
    image: np.ndarray,
    annotation: dict,
    prediction: dict,
    manifest_row: dict,
    records: list[dict],
    output_path: Path,
    allow_point_estimate: bool,
    raw_sign: bool,
    agreement_mode: bool = False,
    evidence: dict | None = None,
) -> dict[str, int]:
    h, w = image.shape[:2]
    by_arc = {}
    by_ring: dict[str, list[np.ndarray]] = {}
    for ring in annotation.get("rings", []):
        ring_id = str(ring.get("ring_id"))
        for arc in ring.get("arcs", []):
            points = np.asarray(arc.get("points_crop_px", []), dtype=float)
            if len(points) < 2:
                continue
            by_arc[str(arc.get("arc_id"))] = points
            by_ring.setdefault(ring_id, []).append(points)

    # Prefer the primary subarc partition. Fall back to ring-level records when
    # a smoke run deliberately omits expensive segment delete-refits.
    partitions = sorted({str(row.get("partition_id")) for row in records})
    subarc = [value for value in partitions if value.startswith("subarc:")]
    selected_partition = subarc[0] if subarc else "ring"
    selected = [row for row in records if str(row.get("partition_id")) == selected_partition]
    if agreement_mode:
        counts = {key: 0 for key in AGREEMENT_COUNTS}
    else:
        counts = {label: 0 for label in CONTRIBUTION_COLORS}

    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    for record in selected:
        if agreement_mode and evidence is None:
            counts["NO_DATA"] += 1
            continue
        if agreement_mode:
            value = agreement_value(
                evidence,
                str(record.get("arc_id")),
                record.get("interval_fraction"),
                prediction.get("raw_center_crop_px"),
            )
            if value is None:
                counts["NO_DATA"] += 1
                continue
            if value >= AGREEMENT_HIGH:
                counts["AGREE_HIGH"] += 1
            elif value >= AGREEMENT_MID:
                counts["AGREE_MID"] += 1
            elif value >= AGREEMENT_LOW:
                counts["DEVIATE"] += 1
            else:
                counts["DEVIATE_STRONG"] += 1
            color = agreement_color(value)
        else:
            label = contribution_label(record, allow_point_estimate, raw_sign)
            counts[label] += 1
            color = CONTRIBUTION_COLORS[label]
        arc_id = record.get("arc_id")
        if arc_id is not None and evidence is not None:
            exact = evidence_segment(evidence, str(arc_id), record.get("interval_fraction"))
            segments = [exact] if exact is not None else []
        elif arc_id is not None and str(arc_id) in by_arc:
            segments = [arc_segment(by_arc[str(arc_id)], record.get("interval_fraction"))]
        else:
            segments = by_ring.get(str(record.get("ring_id")), [])
        for segment in segments:
            ax.plot(segment[:, 0], segment[:, 1], color=color, lw=3.0, alpha=0.92)

    gt = np.asarray(manifest_row["pith_crop_px"], dtype=float)
    center = prediction.get("raw_center_crop_px")
    outside_marker(ax, gt, w, h, "#00A878", "GT")
    if center is not None:
        outside_marker(ax, np.asarray(center, dtype=float), w, h, "#D1495B", "estimate")
    if agreement_mode:
        mode = "geometric agreement with consensus pith"
    elif raw_sign:
        mode = "raw signed delta; below-threshold values included"
    elif allow_point_estimate:
        mode = "point-estimate label; no replay CI"
    else:
        mode = "formal CI label"
    ax.set_title(
        f"{manifest_row['crop_id']}\ncontribution={selected_partition} ({mode})"
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x / column [px]")
    ax.set_ylabel("y / row [px]")
    if agreement_mode:
        legend = [
            Line2D([0], [0], color="#009E73", lw=3, label="points at consensus (>=0.90)"),
            Line2D([0], [0], color=agreement_color(0.80), lw=3, label="mostly aligned (0.70-0.90)"),
            Line2D([0], [0], color=agreement_color(0.55), lw=3, label="deviating (0.40-0.70)"),
            Line2D([0], [0], color="#D55E00", lw=3, label="strongly deviating (<0.40)"),
        ]
    else:
        legend = [
            Line2D([0], [0], color=CONTRIBUTION_COLORS["BENEFICIAL_GT"], lw=3, label="positive / beneficial"),
            Line2D([0], [0], color=CONTRIBUTION_COLORS["HARMFUL_GT"], lw=3, label="negative / harmful"),
            Line2D([0], [0], color=CONTRIBUTION_COLORS["NEUTRAL"], lw=3, label="neutral"),
            Line2D([0], [0], color=CONTRIBUTION_COLORS["UNCERTAIN"], lw=3, label="uncertain"),
        ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    counts["partition"] = selected_partition
    return counts


def render_explain_cards(
    explain_path: Path,
    crop_ids: list[str],
    image_root: Path,
    gt_by_crop: dict[str, list[float]],
    est_by_crop: dict[str, list[float] | None],
    group_by_crop: dict[str, str],
    error_by_crop: dict[str, float | None],
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    """Overlay per-crop sub-arc explanation records colored by category."""
    wanted = set(crop_ids)
    labels = EXPLAIN_LABELS if setup_cjk_font() else EXPLAIN_LABELS_EN
    by_crop: dict[str, list[dict]] = {cid: [] for cid in crop_ids}
    with explain_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cid = str(rec.get("crop_id"))
            if cid in wanted:
                by_crop[cid].append(rec)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, int]] = {}
    for cid in crop_ids:
        records = by_crop.get(cid, [])
        image = cv2.imread(str(image_root / f"{cid}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_root / f"{cid}.png")
        h, w = image.shape[:2]
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        counts: dict[str, int] = {key: 0 for key in EXPLAIN_COLORS}
        for rec in records:
            category = str(rec.get("category") or "MODERATE")
            counts[category] = counts.get(category, 0) + 1
            pts = np.asarray(rec.get("points_px") or [], dtype=float)
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], color=EXPLAIN_COLORS[category], lw=2.2, alpha=0.9)
        gt = gt_by_crop.get(cid)
        if gt is not None:
            outside_marker(ax, gt, w, h, "#00A878", "GT")
        est = est_by_crop.get(cid)
        if est is not None:
            outside_marker(ax, est, w, h, "#D1495B", "estimate")
        group = group_by_crop.get(cid, "?")
        err = error_by_crop.get(cid)
        err_txt = f"{err:.1f}px" if err is not None else "REJECT"
        ax.set_title(f"{cid}\n组={group} error={err_txt}")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xlabel("x / column [px]")
        ax.set_ylabel("y / row [px]")
        legend = [
            Line2D([0], [0], color=color, lw=3, label=labels[label])
            for label, color in EXPLAIN_COLORS.items()
            if counts.get(label, 0) > 0
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=7)
        fig.savefig(output_dir / f"{cid}.png", dpi=160)
        plt.close(fig)
        summaries[cid] = counts
    return summaries


def finite_summary(values: list[float]) -> dict[str, float | int | None]:
    data = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(data):
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None, "rmse": None}
    return {
        "n": int(len(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "max": float(np.max(data)),
        "rmse": float(np.sqrt(np.mean(np.square(data)))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-manifest", required=True)
    ap.add_argument("--result-index", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-cards", type=int, default=100)
    ap.add_argument("--contributions", help="optional directory containing per-crop contribution JSONL")
    ap.add_argument(
        "--point-estimate-contributions",
        action="store_true",
        help="use gt_point_label when formal replay-CI labels abstain; figures are marked preliminary",
    )
    ap.add_argument(
        "--raw-sign-contributions",
        action="store_true",
        help="explicitly show raw signed direction for neutral point estimates (exploratory only)",
    )
    ap.add_argument(
        "--agreement-contributions",
        action="store_true",
        help="show per-subarc geometric pointing agreement with the consensus pith (diagnostic; requires --evidence-root)",
    )
    ap.add_argument(
        "--evidence-root",
        help="optional evidence output root containing per_crop/*.json and *.npz; aligns overlays to sampled nodes",
    )
    ap.add_argument(
        "--explain-records",
        help="subarc_explanations.jsonl from scripts/22_explain_subarcs.py; renders category overlays",
    )
    ap.add_argument(
        "--explain-crops",
        help="comma-separated crop ids to render in explain mode",
    )
    ap.add_argument(
        "--explain-image-root",
        help="directory containing {crop_id}.png grayscale images (grayscale_dataset)",
    )
    ap.add_argument(
        "--explain-manifest-validated",
        help="stage-0 manifest_validated.jsonl for GT markers in explain mode",
    )
    ap.add_argument(
        "--explain-final-results",
        help="stage-7 final_results.jsonl for estimate markers in explain mode",
    )
    ap.add_argument(
        "--explain-grouped-csv",
        help="per_crop_metrics.csv from script 21 for group/error annotations",
    )
    args = ap.parse_args()
    if args.explain_records:
        if not all((args.explain_crops, args.explain_image_root, args.explain_manifest_validated,
                    args.explain_final_results, args.explain_grouped_csv)):
            ap.error("--explain-records requires --explain-crops, --explain-image-root, "
                     "--explain-manifest-validated, --explain-final-results, --explain-grouped-csv")
        import csv as _csv

        crop_ids = [c.strip() for c in args.explain_crops.split(",") if c.strip()]
        gt_by_crop = {str(r["crop_id"]): r["pith_px"] for r in read_jsonl(Path(args.explain_manifest_validated))}
        est_by_crop = {
            str(r["crop_id"]): (r.get("pith_px") if r.get("pith_px") is not None else None)
            for r in read_jsonl(Path(args.explain_final_results))
        }
        group_by_crop: dict[str, str] = {}
        error_by_crop: dict[str, float | None] = {}
        with Path(args.explain_grouped_csv).open(encoding="utf-8-sig", newline="") as fh:
            for row in _csv.DictReader(fh):
                group_by_crop[str(row["crop_id"])] = str(row["group"])
                err = row.get("error_px")
                error_by_crop[str(row["crop_id"])] = float(err) if err not in ("", None) else None
        out = Path(args.output).resolve()
        explain_dir = out / "explain_overlays"
        summaries = render_explain_cards(
            Path(args.explain_records),
            crop_ids,
            Path(args.explain_image_root),
            gt_by_crop,
            est_by_crop,
            group_by_crop,
            error_by_crop,
            explain_dir,
        )
        with (out / "explain_summary.json").open("w", encoding="utf-8") as fh:
            json.dump({"cards": summaries}, fh, ensure_ascii=False, indent=2)
        print("rendered explain overlays:", {cid: counts for cid, counts in summaries.items()})
        return
    if args.agreement_contributions and not args.evidence_root:
        ap.error("--agreement-contributions requires --evidence-root so overlays use the exact sampled nodes")
    manifest = {str(r["crop_id"]): r for r in read_jsonl(Path(args.crop_manifest))}
    result_rows = read_jsonl(Path(args.result_index))
    out = Path(args.output).resolve()
    cards = out / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    contribution_cards = out / "contribution_overlays"
    evidence_root = Path(args.evidence_root).resolve() if args.evidence_root else None
    evidence_cache: dict[str, dict | None] = {}

    def load_evidence(crop_id: str) -> dict | None:
        if evidence_root is None:
            return None
        if crop_id in evidence_cache:
            return evidence_cache[crop_id]
        metadata_path = evidence_root / "per_crop" / f"{crop_id}.json"
        npz_path = evidence_root / "per_crop" / f"{crop_id}.npz"
        if not metadata_path.is_file() or not npz_path.is_file():
            evidence_cache[crop_id] = None
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as arrays:
            bundle = {
                "arc_index": {str(value): index for index, value in enumerate(metadata["arc_ids"])},
                "arc_indices": np.asarray(arrays["arc_index"], dtype=np.int64),
                "arc_fractions": np.asarray(arrays["arc_fraction"], dtype=float),
                "points_crop_px": np.asarray(arrays["points_crop_px"], dtype=float),
                "tangents": np.asarray(arrays["tangents"], dtype=float),
            }
        evidence_cache[crop_id] = bundle
        return bundle
    records = []
    contribution_summaries = {}
    for row in sorted(result_rows, key=lambda r: str(r["crop_id"])):
        cid = str(row["crop_id"])
        if cid not in manifest or row.get("status") != "FINISHED":
            continue
        item = manifest[cid]
        pred = json.loads(Path(row["prediction_path"]).read_text(encoding="utf-8"))
        image = cv2.imread(str(item["crop_image_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(item["crop_image_path"])
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax.imshow(rgb)
        ann = json.loads(Path(item["crop_annotation_path"]).read_text(encoding="utf-8"))
        for ring in ann.get("rings", []):
            for arc in ring.get("arcs", []):
                pts = np.asarray(arc.get("points_crop_px", []), dtype=float)
                if len(pts) > 1:
                    ax.plot(pts[:, 0], pts[:, 1], color="#4C78A8", lw=0.7, alpha=0.65)
        gt = np.asarray(item["pith_crop_px"], dtype=float)
        est = np.asarray(pred.get("raw_center_crop_px"), dtype=float) if pred.get("raw_center_crop_px") is not None else None
        outside_marker(ax, gt, w, h, "#00A878", "GT")
        if est is not None:
            outside_marker(ax, est, w, h, "#D1495B", f"estimate {pred.get('state')}")
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_title(f"{cid}\nclass={item.get('pith_class')} state={pred.get('state')}")
        ax.set_xlabel("x / column [px]"); ax.set_ylabel("y / row [px]")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=8)
        fig.savefig(cards / f"{cid}.png", dpi=150)
        plt.close(fig)
        if args.contributions:
            contribution_path = Path(args.contributions) / f"{cid}.jsonl"
            if contribution_path.is_file():
                contribution_summaries[cid] = render_contribution_card(
                    image,
                    ann,
                    pred,
                    item,
                    read_jsonl(contribution_path),
                    contribution_cards / f"{cid}.png",
                    args.point_estimate_contributions,
                    args.raw_sign_contributions,
                    args.agreement_contributions,
                    load_evidence(cid),
                )
        error = float(np.linalg.norm(est - gt)) if est is not None else math.nan
        diagonal = math.hypot(w, h)
        records.append({
            "crop_id": cid, "tree_id": item.get("tree_id"), "pith_class": item.get("pith_class"),
            "state": pred.get("state"), "model_risk": pred.get("model_risk"),
            "search_adequate": pred.get("search_adequate"), "production_usable": pred.get("production_usable"),
            "gt_x": gt[0], "gt_y": gt[1], "estimate_x": est[0] if est is not None else math.nan,
            "estimate_y": est[1] if est is not None else math.nan, "error_px": error,
            "error_norm": error / diagonal if math.isfinite(error) else math.nan,
            "gt_inside": bool(item.get("pith_inside_crop")),
            "estimate_inside": bool(est is not None and 0 <= est[0] < w and 0 <= est[1] < h),
            "reason_codes": "|".join(pred.get("reason_codes", [])),
        })
    with (out / "smoke_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else ["crop_id"])
        writer.writeheader(); writer.writerows(records)
    by_class = {}
    for cls in sorted({r["pith_class"] for r in records}):
        subset = [r for r in records if r["pith_class"] == cls]
        errs = [r["error_norm"] for r in subset if math.isfinite(r["error_norm"])]
        by_class[cls] = {"n": len(subset), "median_error_norm": float(np.median(errs)) if errs else None,
                         "max_error_norm": float(np.max(errs)) if errs else None,
                         "estimate_inside_rate": float(np.mean([r["estimate_inside"] for r in subset])) if subset else None}
    overall = {
        "error_px": finite_summary([r["error_px"] for r in records]),
        "error_norm": finite_summary([r["error_norm"] for r in records]),
    }
    error_bins = []
    finite_norm = [r["error_norm"] for r in records if math.isfinite(r["error_norm"])]
    for label, lo, hi in ERROR_NORM_BINS:
        count = sum(lo <= value < hi for value in finite_norm)
        error_bins.append({
            "interval": label,
            "lower": lo,
            "upper": None if math.isinf(hi) else hi,
            "count": count,
            "fraction": count / len(finite_norm) if finite_norm else None,
            "percent": 100.0 * count / len(finite_norm) if finite_norm else None,
        })
    with (out / "error_interval_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(error_bins[0]))
        writer.writeheader(); writer.writerows(error_bins)
    summary = {"n_finished": len(records), "overall": overall, "error_norm_intervals": error_bins,
               "contribution_overlays": contribution_summaries, "by_class": by_class,
               "state_counts": {s: sum(r["state"] == s for r in records) for s in sorted({r["state"] for r in records})},
               "all_have_blocking_reason": all(bool(r["reason_codes"]) for r in records)}
    (out / "smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Local grayscale-mask smoke analysis", "", f"- Finished crops: **{len(records)}**", "- Scope: 2 samples each from 图像上/近髓/远髓 (train split)", "- Image-quality weighting: disabled (mask-only input)", "", "## Result judgment", ""]
    md.append("The solver completed all selected crops, but this is not a production pass: the configuration is provisional local smoke, and every prediction carries governance/model-risk/target-domain reason codes.")
    px = overall["error_px"]
    norm = overall["error_norm"]
    md += ["", "## Overall error", "", "| unit | n | mean | median | P90 | max | RMSE |", "|---|---:|---:|---:|---:|---:|---:|",
           f"| px | {px['n']} | {px['mean']:.3f} | {px['median']:.3f} | {px['p90']:.3f} | {px['max']:.3f} | {px['rmse']:.3f} |",
           f"| normalized by crop diagonal | {norm['n']} | {norm['mean']:.5f} | {norm['median']:.5f} | {norm['p90']:.5f} | {norm['max']:.5f} | {norm['rmse']:.5f} |",
           "", "## Normalized-error interval share", "", "| interval | count | share |", "|---|---:|---:|"]
    for row in error_bins:
        md.append(f"| {row['interval']} | {row['count']} | {row['percent']:.1f}% |")
    md += ["", "## Per-class metrics", "", "| class | n | median normalized error | max normalized error | estimate inside rate |", "|---|---:|---:|---:|---:|"]
    for cls, val in by_class.items():
        md.append(f"| {cls} | {val['n']} | {val['median_error_norm']:.4f} | {val['max_error_norm']:.4f} | {val['estimate_inside_rate']:.3f} |")
    md += ["", "## Visual cards", "", "Each PNG overlays blue ring curves, green GT, and red estimated center. Outside-crop GT/estimates are shown with an arrow.", ""]
    for r in records:
        md.append(f"- [{r['crop_id']}.png](cards/{r['crop_id']}.png) — {r['pith_class']}, state={r['state']}, error_norm={r['error_norm']:.4f}")
    if contribution_summaries:
        md += ["", "## Contribution overlays", ""]
        if args.agreement_contributions:
            md.append(
                "These overlays show per-subarc geometric pointing agreement: the mean |cos| "
                "between each segment's sampled normals and the direction to the estimated "
                "(consensus) pith. Green = the segment points at the same position as the other "
                "rings, orange/red = deviating. This is a display diagnostic of ring geometry, "
                "not the formal delete-refit contribution."
            )
        else:
            if args.point_estimate_contributions:
                md.append("These local-smoke colors use point-estimate GT contribution signs because replay confidence intervals were not run. They are exploratory and must not be reported as formal signed-contribution conclusions.")
            if args.raw_sign_contributions:
                md.append("Green/red additionally shows the raw sign of finite delete-refit deltas, including values inside the registered neutral threshold. Color therefore indicates direction, not statistical or practical significance.")
            elif args.point_estimate_contributions:
                md.append("Neutral segments remain gray; only formal beneficial/harmful labels receive green/red. Evidence-node alignment is used when --evidence-root is supplied.")
        for crop_id, values in sorted(contribution_summaries.items()):
            if args.agreement_contributions:
                md.append(
                    f"- [{crop_id}.png](contribution_overlays/{crop_id}.png) — "
                    f"partition={values['partition']}, agree(>=0.90)={values['AGREE_HIGH']}, "
                    f"aligned(0.70-0.90)={values['AGREE_MID']}, deviating(0.40-0.70)={values['DEVIATE']}, "
                    f"strong-deviating(<0.40)={values['DEVIATE_STRONG']}, no-data={values['NO_DATA']}"
                )
            else:
                md.append(
                    f"- [{crop_id}.png](contribution_overlays/{crop_id}.png) — "
                    f"partition={values['partition']}, beneficial={values['BENEFICIAL_GT']}, "
                    f"harmful={values['HARMFUL_GT']}, neutral={values['NEUTRAL']}, "
                    f"raw-positive={values['POSITIVE_RAW']}, raw-negative={values['NEGATIVE_RAW']}, "
                    f"uncertain={values['UNCERTAIN']}"
                )
    (out / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
