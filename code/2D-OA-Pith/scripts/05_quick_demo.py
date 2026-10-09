"""Quick local demo — single image, minimal geometry, no LOAO, no rotation.

Purpose: verify the pipeline works on CPU before sending to server GPU.
"""
from __future__ import annotations

import json, math, os, time
from pathlib import Path
import torch
from PIL import Image

from oapith.types import ArcObservation, CoordinateFrame
from oapith.inference import ProbabilisticGeometryRefiner, RefinerConfig, generate_candidates
from oapith.inference.posterior import build_mode_posteriors
from oapith.calibration import joint_state_probabilities, joint_state_decision, prediction_from_posteriors
from oapith.types import PithState

# ── Edit these ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = Path("D:/教务处实习/木材数据1/patch_dataset/裁剪矩形/T0_B1_N27_A/T0_B1_N27_A_s1000_n0.json")
IMAGE_PATH = Path("D:/教务处实习/木材数据1/patch_dataset/裁剪矩形/T0_B1_N27_A/T0_B1_N27_A_s1000_n0.jpg")

# ── Fast config ───────────────────────────────────────────────
FAST_CONFIG = RefinerConfig(
    sample_spacing=0.015,
    minimum_sigma=0.002,
    student_df=4.0,
    outlier_scale=8.0,
    tangent_scale=0.15,
    switch_radius=2.0,
    near_max_radius=4.0,
    far_allow_affine=False,
    maximum_harmonic=3,
    nesting_margin=0.002,
    warmup_steps=5,
    stage_steps=10,
    em_rounds=1,
    max_modes=2,
    mode_loss_window=12.0,
    deduplication_distance=0.08,
    deduplication_angle=0.02,
    deduplication_inverse_distance=0.02,
    stationarity_tolerance=0.0001,
    disturbance_source_options=(0,),
)


def estimate_tangents(points: torch.Tensor) -> torch.Tensor:
    diffs = points[1:] - points[:-1]
    tangents = (diffs[:-1] + diffs[1:]) / 2.0
    full = torch.cat([diffs[0:1], tangents, diffs[-1:]], dim=0)
    return torch.nn.functional.normalize(full, dim=-1, eps=1e-8)


def main():
    os.chdir(PROJECT_ROOT)

    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Image:  {IMAGE_PATH.name}")
    print(f"JSON:   {JSON_PATH.name}")

    t0 = time.time()

    with open(JSON_PATH, encoding="utf-8") as f:
        ann = json.load(f)
    img = Image.open(IMAGE_PATH)
    w, h = img.size
    frame = CoordinateFrame(w, h)
    print(f"  size={w}x{h}  rings={len(ann['rings'])}  "
          f"gt_pith_local=({ann['cx_local']},{ann['cy_local']})")

    # Build arcs from ring polylines.
    arcs = []
    for ring in ann["rings"]:
        pts_px = torch.tensor(ring["points"], dtype=torch.float64)
        if pts_px.shape[0] < 3:
            continue
        pts_norm = frame.pixel_to_normalized(pts_px)
        tangents = estimate_tangents(pts_norm)
        spacing = torch.linalg.vector_norm(pts_norm[1:] - pts_norm[:-1], dim=-1)
        sigma_val = max(float(spacing.median()) * 0.5, 0.002)
        arcs.append(ArcObservation(
            arc_id=f"r{ring['ring_idx']}",
            ring_id=ring["ring_idx"],
            points=pts_norm,
            tangents=tangents,
            sigma=torch.tensor(sigma_val, dtype=torch.float64),
            prior_reliability=0.95,
        ))
    print(f"  arcs: {len(arcs)}  ({time.time()-t0:.1f}s)")

    # Candidates.
    t1 = time.time()
    candidates = generate_candidates(arcs, neural_candidates=None)
    print(f"  candidates: {len(candidates)}  ({time.time()-t1:.1f}s)")

    # Refine.
    t2 = time.time()
    state_probs = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    refiner = ProbabilisticGeometryRefiner(FAST_CONFIG)
    results, models, packed = refiner.refine(arcs, candidates, state_probabilities=state_probs)
    print(f"  refine: {time.time()-t2:.1f}s")

    # Posteriors.
    posteriors = build_mode_posteriors(results, models, packed)
    if not posteriors:
        print("  ❌ No posteriors")
        return

    best = max(posteriors, key=lambda v: v.weight)
    prediction = prediction_from_posteriors(posteriors, null_probability=0.25)
    joint_probs = joint_state_probabilities(prediction)
    state = joint_state_decision(joint_probs)

    # Results.
    print(f"\n  State: {state.name}  Weight: {best.weight:.3f}  "
          f"Data rank: {best.data.rank}/{best.data.dimension}")

    if best.result.center is not None:
        center_px = frame.normalized_to_pixel(best.result.center)
        err = math.hypot(center_px[0] - ann["cx_local"], center_px[1] - ann["cy_local"])
        print(f"  Pith (px): ({center_px[0]:.0f}, {center_px[1]:.0f})")
        print(f"  GT   (px): ({ann['cx_local']}, {ann['cy_local']})")
        print(f"  Error  px: {err:.0f}")
    elif best.result.direction is not None:
        print(f"  Direction: [{best.result.direction[0]:.3f}, {best.result.direction[1]:.3f}]")
        print(f"  Inv dist:  {best.result.inverse_distance:.6f}")
    else:
        print(f"  (no finite estimate)")

    print(f"\n  Modes ({len(posteriors)}):")
    for i, p in enumerate(sorted(posteriors, key=lambda v: -v.weight)):
        c = frame.normalized_to_pixel(p.result.center) if p.result.center is not None else None
        cs = f"({c[0]:.0f},{c[1]:.0f})" if c is not None else "—"
        print(f"    [{i}] w={p.weight:.3f} center={cs} rank={p.data.rank} mode={p.result.mode.name}")

    print(f"\n  Total: {time.time()-t0:.1f}s ✅")


if __name__ == "__main__":
    main()
