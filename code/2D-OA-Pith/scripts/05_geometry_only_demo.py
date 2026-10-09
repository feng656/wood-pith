"""Geometry-only pith localization demo — uses existing ring annotations, no CNN.

Reads JSON ring annotations, constructs ArcObservation objects, runs the
probabilistic geometry refiner, computes leave-one-arc-out contributions,
and generates green/red contribution overlays.  Also tests rotation
equivariance (0°/90°/180°/270°).

Usage: edit the paths below, then:
    python scripts/05_geometry_only_demo.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from PIL import Image

from oapith.types import ArcObservation, CoordinateFrame
from oapith.inference import (
    ProbabilisticGeometryRefiner,
    RefinerConfig,
    generate_candidates,
)
from oapith.inference.posterior import build_mode_posteriors
from oapith.explain import exact_leave_one_arc_out
from oapith.reporting import save_overlay
from oapith.calibration import (
    joint_state_probabilities,
    joint_state_decision,
    prediction_from_posteriors,
)
from oapith.types import PithState

# ── Edit these paths ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/jzf/woodproject/树髓定位/裁剪矩形")
OUTPUT_DIR = PROJECT_ROOT / "runs" / "geometry_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick a few samples to test.
SAMPLE_NAMES = [
    "T0_B1_N27_A_s1000_n0",   # full patch, 1000px
    "T0_B1_N27_A_s600_n0",    # smaller patch, 600px
    "T0_B1_N32_A_s1000_n0",   # another tree disk
]

# ── Config (mirrors configs/base.yaml geometry section) ────────
GEOMETRY_CONFIG = {
    "sample_spacing": 0.015,
    "minimum_sigma": 0.002,
    "student_df": 4.0,
    "outlier_scale": 8.0,
    "tangent_scale": 0.15,
    "switch_radius": 2.0,
    "near_max_radius": 4.0,
    "far_allow_affine": False,
    "maximum_harmonic": 5,
    "nesting_margin": 0.002,
    "warmup_steps": 35,
    "stage_steps": 50,
    "em_rounds": 3,
    "max_modes": 6,
    "mode_loss_window": 12.0,
    "deduplication_distance": 0.08,
    "deduplication_angle": 0.02,
    "deduplication_inverse_distance": 0.02,
    "stationarity_tolerance": 0.0001,
    "disturbance_source_options": (0, 1),
}


def estimate_tangents(points: torch.Tensor) -> torch.Tensor:
    """Estimate unit tangents from polyline point differences."""
    diffs = points[1:] - points[:-1]
    # Pad to keep same length as points
    first = diffs[0:1]
    last = diffs[-1:]
    padded = torch.cat([first, diffs, last], dim=0)
    # Smooth with 3-point moving average
    tangents = (padded[:-2] + 2 * diffs + padded[2:]) / 4.0
    # Pad first/last
    full = torch.cat([diffs[0:1], tangents, diffs[-1:]], dim=0)
    return torch.nn.functional.normalize(full, dim=-1, eps=1e-8)


def load_annotation(json_path: Path) -> dict:
    """Load a patch annotation JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def annotation_to_arcs(
    ann: dict, frame: CoordinateFrame
) -> list[ArcObservation]:
    """Convert ring polylines from annotation JSON to ArcObservation list."""
    arcs: list[ArcObservation] = []
    for ring in ann.get("rings", []):
        ring_idx = ring["ring_idx"]
        points_px = torch.tensor(ring["points"], dtype=torch.float64)
        if points_px.shape[0] < 3:
            continue  # need at least 3 points for an arc

        # Convert from pixel to normalized coordinates.
        points_norm = frame.pixel_to_normalized(points_px)
        tangents = estimate_tangents(points_norm)

        # Estimate sigma from point spacing (conservative).
        spacing = torch.linalg.vector_norm(
            points_norm[1:] - points_norm[:-1], dim=-1
        )
        median_spacing = float(spacing.median()) * 0.5
        sigma_val = max(median_spacing, 0.002)

        arc = ArcObservation(
            arc_id=f"r{ring_idx}",
            ring_id=ring_idx,
            points=points_norm,
            tangents=tangents,
            sigma=torch.tensor(sigma_val, dtype=torch.float64),
            prior_reliability=0.95,
        )
        arcs.append(arc)
    return arcs


def run_single_image(
    image_path: Path,
    json_path: Path,
    output_dir: Path,
    *,
    do_contributions: bool = True,
) -> dict:
    """Run geometry pipeline on one image and save results."""
    ann = load_annotation(json_path)
    img = Image.open(image_path)
    w, h = img.size

    frame = CoordinateFrame(w, h)
    arcs = annotation_to_arcs(ann, frame)

    result = {
        "sample": ann.get("patch_name", json_path.stem),
        "width": w,
        "height": h,
        "n_arcs": len(arcs),
        "status": "ok",
    }

    if len(arcs) < 2:
        result["status"] = "too_few_arcs"
        print(f"  ⚠ {json_path.stem}: only {len(arcs)} arcs, skipping")
        return result

    # Generate geometry-only candidates (no neural proposals).
    candidates = generate_candidates(arcs, neural_candidates=None)

    # Run geometric optimization.
    config = RefinerConfig(
        sample_spacing=GEOMETRY_CONFIG["sample_spacing"],
        minimum_sigma=GEOMETRY_CONFIG["minimum_sigma"],
        student_df=GEOMETRY_CONFIG["student_df"],
        outlier_scale=GEOMETRY_CONFIG["outlier_scale"],
        tangent_scale=GEOMETRY_CONFIG["tangent_scale"],
        switch_radius=GEOMETRY_CONFIG["switch_radius"],
        near_max_radius=GEOMETRY_CONFIG["near_max_radius"],
        far_allow_affine=GEOMETRY_CONFIG["far_allow_affine"],
        maximum_harmonic=GEOMETRY_CONFIG["maximum_harmonic"],
        nesting_margin=GEOMETRY_CONFIG["nesting_margin"],
        warmup_steps=GEOMETRY_CONFIG["warmup_steps"],
        stage_steps=GEOMETRY_CONFIG["stage_steps"],
        em_rounds=GEOMETRY_CONFIG["em_rounds"],
        max_modes=GEOMETRY_CONFIG["max_modes"],
        mode_loss_window=GEOMETRY_CONFIG["mode_loss_window"],
        deduplication_distance=GEOMETRY_CONFIG["deduplication_distance"],
        deduplication_angle=GEOMETRY_CONFIG["deduplication_angle"],
        deduplication_inverse_distance=GEOMETRY_CONFIG["deduplication_inverse_distance"],
        stationarity_tolerance=GEOMETRY_CONFIG["stationarity_tolerance"],
        disturbance_source_options=GEOMETRY_CONFIG["disturbance_source_options"],
    )

    # Uniform state prior (we don't have a CNN).
    state_probs = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    refiner = ProbabilisticGeometryRefiner(config)
    results, models, packed = refiner.refine(
        arcs, candidates, state_probabilities=state_probs
    )

    posteriors = build_mode_posteriors(results, models, packed)
    if not posteriors:
        result["status"] = "no_posteriors"
        print(f"  ⚠ {json_path.stem}: no valid posteriors")
        return result

    best = max(posteriors, key=lambda v: v.weight)

    # LOAO contributions.
    contributions = None
    if do_contributions and len(arcs) >= 2:
        contributions = exact_leave_one_arc_out(
            arcs,
            refiner,
            best,
            neural_candidates=None,
            state_probabilities=state_probs,
        )

    # State decision.
    prediction = prediction_from_posteriors(
        posteriors,
        null_probability=float(state_probs[int(PithState.NULL)]),
    )
    joint_probs = joint_state_probabilities(prediction)
    state_decision = joint_state_decision(joint_probs)

    # Print summary.
    if best.result.center is not None:
        center_px = frame.normalized_to_pixel(best.result.center).tolist()
        print(f"  ✓ pith=({center_px[0]:.0f}, {center_px[1]:.0f})px "
              f"weight={best.weight:.3f} state={state_decision.name} "
              f"rank={best.data.rank}")
    elif best.result.direction is not None:
        print(f"  → direction=[{best.result.direction[0]:.3f}, "
              f"{best.result.direction[1]:.3f}] "
              f"rho={best.result.inverse_distance:.4f} "
              f"state={state_decision.name}")
    else:
        print(f"  ? state={state_decision.name} (no finite estimate)")

    # Save overlay.
    if state_decision is not PithState.NULL:
        valid_posteriors = [
            v for v in posteriors
            if v.data.valid and v.posterior.valid and v.data.rank > 0
        ]
        if valid_posteriors:
            display = max(valid_posteriors, key=lambda v: v.weight)
            overlay_path = output_dir / f"{json_path.stem}_overlay.png"
            save_overlay(
                str(image_path),
                str(overlay_path),
                arcs,
                frame,
                display,
                contributions,
            )
            print(f"  📷 overlay → {overlay_path.name}")

    # Print contribution summary.
    if contributions:
        info_gains = [
            c.information_gain
            for c in contributions
            if c.information_gain is not None and math.isfinite(c.information_gain)
        ]
        if info_gains:
            print(f"  📊 contributions: {len(contributions)} arcs, "
                  f"info_gain ∈ [{min(info_gains):.3f}, {max(info_gains):.3f}]")
        struct = [c for c in contributions if c.structural_required]
        if struct:
            print(f"  🔴 structural: {len(struct)} arcs")

    result["center_px"] = (
        frame.normalized_to_pixel(best.result.center).tolist()
        if best.result.center is not None
        else None
    )
    result["state"] = state_decision.name
    result["weight"] = best.weight
    return result


def test_rotation(
    image_path: Path,
    json_path: Path,
    output_dir: Path,
    angles: list[int] | None = None,
) -> dict:
    """Test rotation equivariance: rotate image+annotations, infer, compare."""
    if angles is None:
        angles = [0, 90, 180, 270]

    ann = load_annotation(json_path)
    img = Image.open(image_path)
    results = {}

    for angle in angles:
        tag = f"{json_path.stem}_rot{angle}"
        rotated_img = img.rotate(-angle, expand=True, resample=Image.BICUBIC)
        rot_w, rot_h = rotated_img.size

        # Save rotated image temporarily.
        tmp_img = output_dir / f"{tag}.png"
        rotated_img.save(tmp_img)

        # Transform ring points.
        frame_orig = CoordinateFrame(img.size[0], img.size[1])
        frame_rot = CoordinateFrame(rot_w, rot_h)

        rot_rad = math.radians(angle)
        cos_a, sin_a = math.cos(rot_rad), math.sin(rot_rad)
        rot_matrix = torch.tensor(
            [[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float64
        )

        arcs_rot: list[ArcObservation] = []
        for ring in ann.get("rings", []):
            pts_px = torch.tensor(ring["points"], dtype=torch.float64)
            if pts_px.shape[0] < 3:
                continue

            # Normalize → rotate → denormalize to new frame.
            pts_norm = frame_orig.pixel_to_normalized(pts_px)
            pts_rot_norm = pts_norm @ rot_matrix.T
            # Rotated image has different dimensions, so we work in normalized
            # space directly using the rotated frame as reference.
            # For rotation equivariance we compare in normalized space.
            tangents = estimate_tangents(pts_rot_norm)
            spacing = torch.linalg.vector_norm(
                pts_rot_norm[1:] - pts_rot_norm[:-1], dim=-1
            )
            sigma_val = max(float(spacing.median()) * 0.5, 0.002)
            arcs_rot.append(
                ArcObservation(
                    arc_id=f"r{ring['ring_idx']}",
                    ring_id=ring["ring_idx"],
                    points=pts_rot_norm,
                    tangents=tangents,
                    sigma=torch.tensor(sigma_val, dtype=torch.float64),
                    prior_reliability=0.95,
                )
            )

        if len(arcs_rot) < 2:
            results[angle] = {"status": "too_few_arcs"}
            continue

        candidates = generate_candidates(arcs_rot, neural_candidates=None)
        config = RefinerConfig(**{
            k: v for k, v in GEOMETRY_CONFIG.items()
            if k != "disturbance_source_options"
        })
        config_dict = config.__dict__ if hasattr(config, '__dict__') else {}
        # Work with the values we have.

        state_probs = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
        refiner = ProbabilisticGeometryRefiner(
            RefinerConfig(
                sample_spacing=GEOMETRY_CONFIG["sample_spacing"],
                minimum_sigma=GEOMETRY_CONFIG["minimum_sigma"],
                student_df=GEOMETRY_CONFIG["student_df"],
                outlier_scale=GEOMETRY_CONFIG["outlier_scale"],
                tangent_scale=GEOMETRY_CONFIG["tangent_scale"],
                switch_radius=GEOMETRY_CONFIG["switch_radius"],
                near_max_radius=GEOMETRY_CONFIG["near_max_radius"],
                far_allow_affine=GEOMETRY_CONFIG["far_allow_affine"],
                maximum_harmonic=GEOMETRY_CONFIG["maximum_harmonic"],
                nesting_margin=GEOMETRY_CONFIG["nesting_margin"],
                warmup_steps=GEOMETRY_CONFIG["warmup_steps"],
                stage_steps=GEOMETRY_CONFIG["stage_steps"],
                em_rounds=GEOMETRY_CONFIG["em_rounds"],
                max_modes=GEOMETRY_CONFIG["max_modes"],
                mode_loss_window=GEOMETRY_CONFIG["mode_loss_window"],
                deduplication_distance=GEOMETRY_CONFIG["deduplication_distance"],
                deduplication_angle=GEOMETRY_CONFIG["deduplication_angle"],
                deduplication_inverse_distance=GEOMETRY_CONFIG["deduplication_inverse_distance"],
                stationarity_tolerance=GEOMETRY_CONFIG["stationarity_tolerance"],
                disturbance_source_options=GEOMETRY_CONFIG["disturbance_source_options"],
            )
        )
        ref_results, _, _ = refiner.refine(
            arcs_rot, candidates, state_probabilities=state_probs
        )
        posteriors = build_mode_posteriors(ref_results, None, None)  # simplified
        if posteriors:
            best = max(posteriors, key=lambda v: v.weight)
            if best.result.center is not None:
                # Rotate center back to original orientation.
                center_back = best.result.center @ rot_matrix
                results[angle] = {
                    "center_norm": center_back.tolist(),
                    "weight": best.weight,
                }
            else:
                results[angle] = {
                    "direction": (
                        best.result.direction.tolist()
                        if best.result.direction is not None
                        else None
                    ),
                    "weight": best.weight,
                }
        else:
            results[angle] = {"status": "no_posteriors"}

        # Clean up temp image.
        tmp_img.unlink(missing_ok=True)

    # Compute rotation consistency (E6 metric).
    centers = []
    for angle, r in results.items():
        if "center_norm" in r:
            c = torch.tensor(r["center_norm"])
            centers.append((angle, c))

    if len(centers) >= 2:
        all_centers = torch.stack([c for _, c in centers])
        mean_center = all_centers.mean(0)
        deviations = {
            angle: float(torch.norm(c - mean_center))
            for angle, c in centers
        }
        max_dev = max(deviations.values())
        print(f"\n  🔄 Rotation equivariance (max deviation): {max_dev:.4f} norm units")
        for angle, dev in deviations.items():
            print(f"     {angle}°: {dev:.4f}")
        results["_equivariance"] = {
            "max_deviation": max_dev,
            "per_angle": deviations,
        }

    return results


def main() -> None:
    os.chdir(PROJECT_ROOT)

    for sample_name in SAMPLE_NAMES:
        # Find the sample directory.
        parts = sample_name.rsplit("_", 2)  # e.g. T0_B1_N27_A / s1000 / n0
        sample_dir = "_".join(parts[:-2]) if len(parts) > 2 else sample_name

        json_path = DATA_ROOT / sample_dir / f"{sample_name}.json"
        image_path = DATA_ROOT / sample_dir / f"{sample_name}.jpg"

        if not json_path.exists():
            print(f"\n❌ Missing: {json_path}")
            continue
        if not image_path.exists():
            print(f"\n❌ Missing: {image_path}")
            continue

        sample_out = OUTPUT_DIR / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"📷 {sample_name}")
        print(f"{'='*60}")

        # Basic inference + contributions.
        run_single_image(image_path, json_path, sample_out, do_contributions=True)

        # Rotation test.
        print(f"  🔄 Testing rotations...")
        rot_results = test_rotation(image_path, json_path, sample_out)

        # Save combined results.
        with open(sample_out / "results.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in rot_results.items() if not k.startswith("_")},
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    print(f"\n✅ Done. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
