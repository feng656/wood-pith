from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from oapith.calibration import (
    ConformalCalibrator,
    joint_state_decision,
    joint_state_probabilities,
    prediction_from_posteriors,
)
from oapith.config import load_config
from oapith.contracts import DEFAULT_IMAGE_SIZE, validate_checkpoint_contract
from oapith.explain import exact_leave_one_arc_out
from oapith.geometry.coordinates import far_to_cartesian, normalized_sampling_grid
from oapith.inference import ProbabilisticGeometryRefiner, RefinerConfig, generate_candidates
from oapith.inference.posterior import build_mode_posteriors
from oapith.inference.bootstrap import run_arc_block_bootstrap
from oapith.inference.tiling import (
    tile_distance_truncation_in_global_chart,
    tiled_dense_prediction,
)
from oapith.inference.vectorize import extract_arcs
from oapith.models import OAPithNet
from oapith.reporting import save_overlay, save_prediction_json
from oapith.types import CoordinateFrame, PithState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer an in/out-of-frame pith posterior")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--overlay", help="Optional contribution overlay PNG")
    parser.add_argument("--calibrator")
    parser.add_argument("--device")
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Allow an unverifiable checkpoint without the coordinate/model contract",
    )
    parser.add_argument("--mm-per-pixel", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--exact-contributions", action="store_true")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=0,
        help="Correlated annotation-curve bootstrap draws (expensive)",
    )
    parser.add_argument(
        "--bootstrap-correlation",
        type=float,
        default=0.92,
        help="AR(1) normal-error correlation along each arc",
    )
    return parser


def _read(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _standardize(image: torch.Tensor, data_config: dict[str, object]) -> torch.Tensor:
    mean = image.new_tensor(data_config.get("mean", [0.5, 0.5, 0.5])).view(3, 1, 1)
    std = image.new_tensor(data_config.get("std", [0.25, 0.25, 0.25])).view(3, 1, 1)
    return (image - mean) / std


def _context(
    image: torch.Tensor, frame: CoordinateFrame, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    grid, valid = normalized_sampling_grid(frame, size, device=image.device, dtype=image.dtype)
    square = functional.grid_sample(
        image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return square, valid


def _neural_proposals(output: dict[str, torch.Tensor], switch_radius: float) -> torch.Tensor:
    """Deterministic sigma-point proposals from every learned mixture component.

    These are optimizer starts, not independent evidence: their logits and scales
    are trained on the same image that produced the ring observations.
    """
    proposals: list[torch.Tensor] = []
    near_order = torch.argsort(output["near_logits"][0], descending=True)
    for index in near_order.tolist():
        mean = output["near_mean"][0, index].detach().cpu().double()
        cholesky = output["near_cholesky"][0, index].detach().cpu().double()
        proposals.append(mean)
        for axis in range(2):
            proposals.extend([mean + cholesky[:, axis], mean - cholesky[:, axis]])

    far_order = torch.argsort(output["far_logits"][0], descending=True)
    for index in far_order.tolist():
        direction = output["far_direction"][0, index].detach().cpu().double()
        angle = torch.atan2(direction[1], direction[0])
        angular_sigma = output["far_kappa"][0, index].detach().cpu().double().rsqrt()
        log_eta = output["far_log_eta_mean"][0, index].detach().cpu().double()
        log_eta_sigma = output["far_log_eta_scale"][0, index].detach().cpu().double()
        for angle_offset, radial_offset in (
            (0.0, 0.0),
            (angular_sigma, 0.0),
            (-angular_sigma, 0.0),
            (0.0, log_eta_sigma),
            (0.0, -log_eta_sigma),
        ):
            proposal_direction = torch.stack(
                [torch.cos(angle + angle_offset), torch.sin(angle + angle_offset)]
            )
            eta = torch.exp(log_eta + radial_offset).clamp(1e-5, 1.0)
            proposals.append(
                far_to_cartesian(proposal_direction, eta, switch_radius)
            )
    infinity_direction = output["infinity_direction"][0].detach().cpu().double()
    infinity_angle = torch.atan2(infinity_direction[1], infinity_direction[0])
    infinity_sigma = output["infinity_kappa"][0].detach().cpu().double().rsqrt()
    for angle_offset in (0.0, infinity_sigma, -infinity_sigma):
        direction = torch.stack(
            [
                torch.cos(infinity_angle + angle_offset),
                torch.sin(infinity_angle + angle_offset),
            ]
        )
        for distance in (8.0, 32.0, 64.0):
            proposals.append(direction * distance)
    return torch.stack(proposals)


def run_inference(
    config_path: str | Path,
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    *,
    overlay_path: str | Path | None = None,
    calibrator_path: str | Path | None = None,
    device: str | torch.device | None = None,
    allow_legacy_checkpoint: bool = False,
    mm_per_pixel: tuple[float, float] | list[float] | None = None,
    exact_contributions: bool = False,
    bootstrap_replicates: int = 0,
    bootstrap_correlation: float = 0.92,
) -> None:
    """Run the full image-to-posterior pipeline from Python."""
    config = load_config(config_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = OAPithNet(**config.get("model", {})).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configured_switch = config.get("geometry", {}).get("switch_radius", 2.0)
    validate_checkpoint_contract(
        checkpoint,
        model=model,
        run_config=config,
        switch_radius=float(configured_switch),
        allow_legacy=allow_legacy_checkpoint,
    )
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    raw = _read(image_path)
    spacing = mm_per_pixel or [None, None]
    frame = CoordinateFrame(raw.shape[-1], raw.shape[-2], spacing[0], spacing[1])
    standardized = _standardize(raw, config["data"]).unsqueeze(0).to(device)
    raw_valid = torch.ones(1, 1, raw.shape[-2], raw.shape[-1], device=device)
    inference_config = config.get("inference", {})
    training_image_size = int(config["data"].get("image_size", DEFAULT_IMAGE_SIZE))
    tile_size = int(inference_config.get("tile_size", training_image_size))
    if tile_size != training_image_size:
        raise ValueError(
            "inference.tile_size must equal training data.image_size so the dense "
            "uncertainty/receptive-field contract is preserved"
        )
    dense = tiled_dense_prediction(
        model,
        standardized,
        tile_size=tile_size,
        overlap=int(inference_config.get("overlap", 192)),
        batch_size=int(inference_config.get("tile_batch_size", 2)),
        valid_mask=raw_valid,
    )
    context, context_valid = _context(
        standardized.squeeze(0), frame, training_image_size
    )
    with torch.inference_mode():
        global_output = model(context, context_valid)
    arcs = extract_arcs(
        torch.sigmoid(dense["ring_logits"][0]),
        dense["orientation"][0],
        dense["log_variance"][0],
        dense["embedding"][0],
        threshold=float(inference_config.get("ring_threshold", 0.35)),
        minimum_pixels=int(inference_config.get("minimum_arc_pixels", 12)),
        sample_spacing=float(config.get("geometry", {}).get("sample_spacing", 0.015)),
        # Dense uncertainty is predicted in the normalized *tile* chart. Convert
        # it to the global image chart before geometric inference.
        distance_truncation=tile_distance_truncation_in_global_chart(
            float(config["data"].get("raster_truncation", 0.04)),
            tile_size,
            frame,
        ),
        frame=frame,
    )
    state_probabilities = torch.softmax(global_output["state_logits"][0], -1).detach().cpu()
    if not arcs:
        value = {
            "status": "rejected",
            "reason": "no usable ring arcs",
            "neural_state_probabilities": state_probabilities.tolist(),
            "joint_state_probabilities": [0.0, 0.0, 0.0, 1.0],
            "state_decision": "null",
            "output_type": "null",
            "coordinate_frame": frame.as_dict(),
        }
        Path(output_path).write_text(json.dumps(value, indent=2), encoding="utf-8")
        if overlay_path:
            Image.open(image_path).convert("RGB").save(overlay_path)
        return

    switch_radius = float(config.get("geometry", {}).get("switch_radius", 2.0))
    neural_candidates = _neural_proposals(global_output, switch_radius)
    candidates = generate_candidates(arcs, neural_candidates=neural_candidates)
    geometry_values = dict(config.get("geometry", {}))
    if "disturbance_source_options" in geometry_values:
        geometry_values["disturbance_source_options"] = tuple(
            geometry_values["disturbance_source_options"]
        )
    refiner = ProbabilisticGeometryRefiner(RefinerConfig(**geometry_values))
    results, models, packed = refiner.refine(
        arcs, candidates, state_probabilities=state_probabilities
    )
    posteriors = build_mode_posteriors(results, models, packed)
    best = max(posteriors, key=lambda value: value.weight)
    contributions = None
    if len(arcs) >= 2 and (
        exact_contributions
        or bool(inference_config.get("exact_arc_contribution", False))
    ):
        contributions = exact_leave_one_arc_out(
            arcs,
            refiner,
            best,
            neural_candidates=neural_candidates,
            state_probabilities=state_probabilities,
        )
    prediction = prediction_from_posteriors(
        posteriors,
        null_probability=float(state_probabilities[int(PithState.NULL)]),
    )
    joint_probabilities = joint_state_probabilities(prediction)
    bootstrap = None
    if bootstrap_replicates > 0:
        bootstrap = run_arc_block_bootstrap(
            arcs,
            refiner,
            replicates=bootstrap_replicates,
            correlation=bootstrap_correlation,
            neural_candidates=neural_candidates,
            state_probabilities=state_probabilities,
        )
    conformal_value: dict[str, object] = {"prediction": asdict(prediction)}
    if calibrator_path:
        calibrator = ConformalCalibrator.load(calibrator_path)
        threshold = calibrator.threshold
        conformal_value.update(
            {
                "alpha": calibrator.alpha,
                "threshold": threshold
                if threshold is not None and math.isfinite(threshold)
                else None,
                "threshold_is_infinite": threshold is not None and math.isinf(threshold),
                "groups": calibrator.number_groups,
            }
        )
    save_prediction_json(
        output_path,
        posteriors,
        frame,
        contributions=contributions,
        conformal=conformal_value,
        bootstrap=bootstrap,
        neural_state_probabilities=state_probabilities.tolist(),
        joint_state_probabilities=joint_probabilities,
        prediction=prediction,
    )
    if overlay_path:
        valid_posteriors = [
            value
            for value in posteriors
            if value.data.valid
            and value.posterior.valid
            and value.data.rank > 0
        ]
        if (
            joint_state_decision(joint_probabilities) is PithState.NULL
            or not valid_posteriors
        ):
            Image.open(image_path).convert("RGB").save(overlay_path)
        else:
            display = max(valid_posteriors, key=lambda value: value.weight)
            save_overlay(image_path, overlay_path, arcs, frame, display, contributions)


def main() -> None:
    args = build_parser().parse_args()
    run_inference(
        args.config,
        args.checkpoint,
        args.image,
        args.output,
        overlay_path=args.overlay,
        calibrator_path=args.calibrator,
        device=args.device,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        mm_per_pixel=args.mm_per_pixel,
        exact_contributions=args.exact_contributions,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_correlation=args.bootstrap_correlation,
    )


if __name__ == "__main__":
    main()
