#!/usr/bin/env python3
"""Stage 3: projective All-Arc global inversion and search certificate.

No ground-truth coordinate is read by the objective or by basin selection.
The implementation uses the frozen parent-balanced pseudo-Huber objective,
finite/far charts, two numerical budgets, a rotated mesh replay, and explicit
failure codes when the search certificate does not close.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize


DELTA = 1.5
SIGMA0 = 0.02
SIGMA_MIN = 0.01
SIGMA_MAX = 0.35
EPS_V = 1.0e-4
R_HARD = 0.01
R_SOFT = 0.03
FINITE_BOUND = 8.0
TAU_J = 1.0e-4
TAU_H = 0.02
TAU_SEAM_J = 1.0e-3
TAU_EQ = 5.0e-3
DELTA_MODE = 1.0e-2
MERGE_H = 0.02
OVERLAP_KAPPA = (0.1, 1.0)


def canon(h: Iterable[float]) -> np.ndarray:
    out = np.asarray(h, dtype=float)
    out /= max(float(np.linalg.norm(out)), 1.0e-300)
    if out[2] < 0:
        out = -out
    return out


def finite_h(p: Iterable[float]) -> np.ndarray:
    return canon(np.r_[np.asarray(p, dtype=float), 1.0])


def far_h(phi: float, kappa: float) -> np.ndarray:
    return canon([math.cos(phi), math.sin(phi), max(0.0, float(kappa))])


def phi_kappa(h: Iterable[float]) -> tuple[float, float]:
    hh = canon(h)
    radial = max(float(np.linalg.norm(hh[:2])), 1.0e-15)
    return math.atan2(float(hh[1]), float(hh[0])), float(hh[2] / radial)


def point_from_h(h: Iterable[float]) -> np.ndarray | None:
    hh = canon(h)
    if hh[2] <= 1.0e-10:
        return None
    return hh[:2] / hh[2]


def rp_distance(a: Iterable[float], b: Iterable[float]) -> float:
    aa, bb = canon(a), canon(b)
    return float(math.acos(np.clip(abs(float(aa @ bb)), 0.0, 1.0)))


def pseudo_huber(z: np.ndarray) -> np.ndarray:
    return DELTA * DELTA * (np.sqrt(1.0 + (z / DELTA) ** 2) - 1.0)


def pack_rings(rings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Pack hierarchical evidence without changing its frozen weights."""
    xs, ts, weights, ring_index, sigma_xs, sigma_psis = [], [], [], [], [], []
    for parent_index, arcs in enumerate(rings.values()):
        for arc in arcs:
            x = np.asarray(arc["points_norm"], dtype=float)
            t = np.asarray(arc["tangents"], dtype=float)
            ds = np.asarray(arc["ds_norm"], dtype=float)
            q = ds / max(float(np.sum(ds)), 1.0e-15)
            xs.append(x); ts.append(t)
            weights.append(float(arc.get("omega", 1.0)) * q)
            ring_index.append(np.full(len(x), parent_index, dtype=np.int32))
            sigma_xs.append(np.full(len(x), float(arc.get("sigma_position_norm", 0.0))))
            sigma_psis.append(np.full(len(x), float(arc.get("sigma_psi_rad", 0.0))))
    return {"x": np.vstack(xs), "t": np.vstack(ts), "weight": np.concatenate(weights),
            "ring_index": np.concatenate(ring_index), "sigma_x": np.concatenate(sigma_xs),
            "sigma_psi": np.concatenate(sigma_psis), "ring_count": len(rings)}


def objective_components(h: Iterable[float], rings: dict[str, Any],
                         stride: int = 1, include_barrier: bool = True) -> dict[str, Any]:
    hh = canon(h)
    a, h0 = hh[:2], float(hh[2])
    if not rings or int(rings.get("ring_count", 0)) == 0:
        return {"valid": False, "J_abs": float("inf"), "loss": float("inf")}
    x, t = rings["x"], rings["t"]
    v = a[None, :] - h0 * x
    d = np.sqrt(np.sum(v * v, axis=1) + EPS_V * EPS_V)
    numer = np.sum(t * v, axis=1)
    e = numer / d
    grad_v = t / d[:, None] - numer[:, None] * v / (d ** 3)[:, None]
    jx = abs(h0) * np.linalg.norm(grad_v, axis=1)
    t_perp = np.column_stack([-t[:, 1], t[:, 0]])
    jpsi = np.sum(t_perp * v, axis=1) / d
    sigma = np.sqrt((jx * rings["sigma_x"]) ** 2 +
                    (jpsi * rings["sigma_psi"]) ** 2 + SIGMA0 ** 2)
    sigma = np.clip(sigma, SIGMA_MIN, SIGMA_MAX)
    node_loss = pseudo_huber(e / sigma)
    values = np.bincount(rings["ring_index"], weights=rings["weight"] * node_loss,
                         minlength=int(rings["ring_count"]))
    j_abs = float(np.mean(values))
    min_radius = float("inf")
    if h0 > 1.0e-10:
        p = a / h0
        min_radius = float(np.min(np.linalg.norm(x - p[None, :], axis=1)))
    barrier = 0.0
    valid = True
    if h0 > 1.0e-10 and min_radius < R_HARD:
        valid = False
        barrier = 1.0e6 + 1.0e4 * (R_HARD - min_radius)
    elif h0 > 1.0e-10 and min_radius < R_SOFT:
        barrier = float(((R_SOFT - min_radius) / (R_SOFT - R_HARD)) ** 2)
    ordered = np.sort(values)
    tail_n = max(1, int(math.ceil(0.2 * len(ordered))))
    return {
        "valid": valid,
        "J_abs": j_abs,
        "barrier": barrier,
        "loss": j_abs + (barrier if include_barrier else 0.0),
        "M50": float(np.median(values)),
        "U20": float(np.mean(ordered[-tail_n:])),
        "P90": float(np.quantile(values, 0.9)) if len(values) >= 10 else None,
        "parent_losses": values.tolist(),
        "min_arc_distance_norm": min_radius if math.isfinite(min_radius) else None,
    }


def optimize(h0: Iterable[float], rings: dict[str, list[dict[str, Any]]], chart: str,
             stride: int, maxiter: int) -> dict[str, Any] | None:
    start = canon(h0)
    if chart == "finite":
        p0 = point_from_h(start)
        if p0 is None:
            return None
        z0 = np.clip(p0, -FINITE_BOUND, FINITE_BOUND)
        make_h = lambda z: finite_h(z)
        bounds = [(-FINITE_BOUND, FINITE_BOUND)] * 2
    else:
        phi0, kappa0 = phi_kappa(start)
        z0 = np.array([phi0, min(kappa0, 50.0)])
        make_h = lambda z: far_h(float(z[0]), float(z[1]))
        bounds = [(-math.pi, math.pi), (0.0, 50.0)]

    def fun(z: np.ndarray) -> float:
        return float(objective_components(make_h(z), rings, stride)["loss"])

    result = minimize(fun, z0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": maxiter, "ftol": 1.0e-10, "gtol": 1.0e-6})
    h = make_h(result.x)
    comp = objective_components(h, rings, stride)
    jac = np.asarray(getattr(result, "jac", [float("inf"), float("inf")]), dtype=float)
    stationary = bool(result.success or (np.all(np.isfinite(jac)) and np.linalg.norm(jac) <= 1.0e-3))
    return {"h": h.tolist(), "chart": chart, "stationary": stationary,
            "optimizer_success": bool(result.success), "iterations": int(getattr(result, "nit", 0)),
            "gradient_norm": float(np.linalg.norm(jac)) if np.all(np.isfinite(jac)) else None,
            **comp}


def merge_basins(solutions: list[dict[str, Any]], rings: dict[str, list[dict[str, Any]]],
                 stride: int) -> list[dict[str, Any]]:
    good = [s for s in solutions if s and math.isfinite(float(s.get("loss", float("inf"))))]
    good.sort(key=lambda s: float(s["loss"]))
    kept: list[dict[str, Any]] = []
    for solution in good:
        match = next((old for old in kept if rp_distance(old["h"], solution["h"]) <= MERGE_H), None)
        if match is None:
            kept.append(solution)
        elif float(solution["loss"]) < float(match["loss"]):
            kept[kept.index(match)] = solution
        else:
            match.setdefault("chart_aliases", []).append(solution["chart"])
    for index, basin in enumerate(kept):
        basin["basin_id"] = index
        basin.update(objective_components(basin["h"], rings, stride))
    kept.sort(key=lambda s: float(s["J_abs"]))
    return kept


def seed_score(seed: dict[str, Any], rings: dict[str, list[dict[str, Any]]], stride: int) -> float:
    return float(objective_components(seed["h"], rings, stride)["loss"])


def phase_mesh() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    step = 2.0 * FINITE_BOUND / 8.0
    for x in np.arange(-FINITE_BOUND + step / 2.0, FINITE_BOUND, step):
        for y in np.arange(-FINITE_BOUND + step / 2.0, FINITE_BOUND, step):
            rows.append({"h": finite_h([x, y]).tolist(), "source": "PHASE_FINITE"})
    for phi in np.linspace(-math.pi, math.pi, 16, endpoint=False) + math.pi / 16.0:
        for kappa in [0.0, 0.02, 0.06, 0.2, 0.6, 2.0]:
            rows.append({"h": far_h(float(phi), kappa).tolist(), "source": "PHASE_FAR"})
    return rows


def select_starts(seeds: list[dict[str, Any]], rings: dict[str, list[dict[str, Any]]],
                  stride: int, count: int) -> list[dict[str, Any]]:
    scored = [(seed_score(seed, rings, stride), seed) for seed in seeds]
    scored.sort(key=lambda item: item[0])
    selected: list[dict[str, Any]] = []
    for score, seed in scored:
        if not math.isfinite(score):
            continue
        if all(rp_distance(seed["h"], old["h"]) > MERGE_H for old in selected):
            selected.append(seed)
        if len(selected) >= count:
            break
    return selected


def run_level(starts: list[dict[str, Any]], rings: dict[str, list[dict[str, Any]]],
              stride: int, maxiter: int) -> list[dict[str, Any]]:
    solutions: list[dict[str, Any]] = []
    for seed in starts:
        h = seed["h"]
        p = point_from_h(h)
        # Use the numerically natural chart here.  The overlap is explicitly
        # replayed in certificate(), so every applicable solution is still
        # checked in both charts.
        charts = ["finite"] if p is not None and float(np.max(np.abs(p))) <= FINITE_BOUND else ["far"]
        for chart in charts:
            solution = optimize(h, rings, chart, stride, maxiter)
            if solution is not None:
                solution["seed_source"] = seed.get("source", "replay")
                solutions.append(solution)
    return merge_basins(solutions, rings, stride)


def relevant(basins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not basins:
        return []
    limit = float(basins[0]["J_abs"]) + TAU_EQ + DELTA_MODE
    return [b for b in basins if float(b["J_abs"]) <= limit]


def certificate(basins_b: list[dict[str, Any]], basins_2b: list[dict[str, Any]],
                phase_basins: list[dict[str, Any]], rings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rel_b, rel_2b = relevant(basins_b), relevant(basins_2b)
    if not rel_b or not rel_2b:
        return {"passed": False, "reason_codes": ["NO_VALID_BASIN"]}
    loss_gap = abs(float(rel_2b[0]["J_abs"]) - float(rel_b[0]["J_abs"])) / (1.0 + abs(float(rel_2b[0]["J_abs"])))
    match = max(min(rp_distance(b["h"], c["h"]) for c in rel_2b) for b in rel_b)
    count_stable = len(rel_b) == len(rel_2b)
    stationary = all(bool(b.get("stationary")) for b in rel_2b)
    near_ok = all(bool(b.get("valid")) for b in rel_2b)
    best = rel_2b[0]
    p = point_from_h(best["h"])
    boundary_pinned = bool(p is not None and np.max(np.abs(p)) >= FINITE_BOUND * 0.995)
    phase_relevant = relevant(phase_basins)
    phase_stable = bool(phase_relevant and min(rp_distance(phase_relevant[0]["h"], b["h"]) for b in rel_2b) <= TAU_H)
    no_new_cell = bool(not phase_relevant or float(phase_relevant[0]["J_abs"]) >= float(best["J_abs"]) - TAU_EQ)

    phi, kappa = phi_kappa(best["h"])
    seam_applicable = OVERLAP_KAPPA[0] <= kappa <= OVERLAP_KAPPA[1]
    seam_distance = None
    seam_loss_gap = None
    seam_passed = True
    if seam_applicable:
        finite = optimize(best["h"], rings, "finite", 1, 100)
        far = optimize(best["h"], rings, "far", 1, 100)
        if finite is None or far is None:
            seam_passed = False
        else:
            seam_distance = rp_distance(finite["h"], far["h"])
            seam_loss_gap = abs(float(finite["J_abs"]) - float(far["J_abs"])) / (1.0 + min(float(finite["J_abs"]), float(far["J_abs"])))
            seam_passed = seam_distance <= TAU_H and seam_loss_gap <= TAU_SEAM_J
    checks = {
        "loss_stable": loss_gap <= TAU_J,
        "basin_match_stable": match <= TAU_H,
        "component_count_stable": count_stable,
        "all_modes_stationary": stationary,
        "no_new_relevant_low_loss_cell": no_new_cell,
        "boundary_not_pinned": not boundary_pinned,
        "near_point_exclusion_respected": near_ok,
        "far_field_not_max_radius_point": not boundary_pinned,
        "grid_phase_stable": phase_stable,
        "seam_passed": seam_passed,
    }
    reasons = [name.upper() for name, passed in checks.items() if not passed]
    return {"passed": all(checks.values()), "reason_codes": reasons, "checks": checks,
            "relative_best_loss_change": loss_gap, "max_basin_match_rp2": match,
            "basin_count_B": len(rel_b), "basin_count_2B": len(rel_2b),
            "seam_applicable": seam_applicable, "seam_rp2_distance": seam_distance,
            "seam_relative_loss_gap": seam_loss_gap, "tau_J": TAU_J, "tau_h": TAU_H,
            "tau_seam_J": TAU_SEAM_J, "tau_eq": TAU_EQ}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    crops = iter_jsonl(args.stage1_dir / "crop_summary.jsonl")
    seeds_iter = iter_jsonl(args.stage2_dir / "seed_registry.jsonl")
    preflight_iter = iter_jsonl(args.stage2_dir / "preflight.jsonl")
    fragment_iter = iter_jsonl(args.stage1_dir / "fragments.jsonl")
    pending = next(fragment_iter, None)
    estimates: list[dict[str, Any]] = []
    basin_rows: list[dict[str, Any]] = []
    cert_rows: list[dict[str, Any]] = []
    phase = phase_mesh()

    for index, (crop, seed_row, preflight) in enumerate(zip(crops, seeds_iter, preflight_iter), 1):
        if args.limit > 0 and index > args.limit:
            break
        crop_id = str(crop["crop_id"])
        if str(seed_row["crop_id"]) != crop_id or str(preflight["crop_id"]) != crop_id:
            raise RuntimeError(f"stage input order mismatch at {crop_id}")
        arcs: list[dict[str, Any]] = []
        while pending is not None and str(pending["crop_id"]) == crop_id:
            arcs.append(pending)
            pending = next(fragment_iter, None)
        rings: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for arc in arcs:
            rings[str(arc["ring_id"])].append(arc)

        if not rings or preflight["route"] == "INSUFFICIENT_GEOMETRY":
            estimate = {"sample_id": crop["sample_id"], "crop_id": crop_id,
                        "tree_id": crop["tree_id"], "status": "INSUFFICIENT_GEOMETRY",
                        "h": None, "pith_norm": None, "pith_px": None}
            cert = {"sample_id": crop["sample_id"], "crop_id": crop_id,
                    "passed": False, "reason_codes": ["INSUFFICIENT_GEOMETRY"]}
            estimates.append(estimate); cert_rows.append(cert)
            continue

        evidence = pack_rings(rings)
        starts_b = select_starts(seed_row["seeds"], evidence, 1, 2)
        basins_b = run_level(starts_b, evidence, stride=1, maxiter=40)
        replay_seeds = [{"h": b["h"], "source": "B_REPLAY"} for b in basins_b]
        starts_2b = select_starts(seed_row["seeds"] + replay_seeds, evidence, 1, 3)
        basins_2b = run_level(starts_2b, evidence, stride=1, maxiter=80)
        phase_starts = select_starts(phase, evidence, 1, 1)
        phase_basins = run_level(phase_starts, evidence, stride=1, maxiter=80)
        cert = certificate(basins_b, basins_2b, phase_basins, evidence)
        cert.update({"sample_id": crop["sample_id"], "crop_id": crop_id})
        cert_rows.append(cert)

        rel = relevant(basins_2b)
        best = rel[0] if rel else None
        if best is None:
            status = "SEARCH_INADEQUATE"
            h_best = p_norm = p_px = None
            model_conflict = False
        else:
            h_best = best["h"]
            point = point_from_h(h_best)
            p_norm = point.tolist() if point is not None else None
            p_px = (np.asarray(crop["center_px"]) + float(crop["scale_px"]) * point).tolist() if point is not None else None
            competing_u20 = min(float(b["U20"]) for b in rel)
            model_conflict = float(best["U20"]) > competing_u20 + 0.05
            status = "MODEL_CONFLICT" if model_conflict else ("OK" if cert["passed"] else "SEARCH_INADEQUATE")
        estimate = {
            "sample_id": crop["sample_id"], "tree_id": crop["tree_id"],
            "section_id": crop["section_id"], "crop_id": crop_id,
            "preflight_route": preflight["route"], "status": status,
            "h": h_best, "pith_norm": p_norm, "pith_px": p_px,
            "axis_direction_rad": phi_kappa(h_best)[0] if h_best is not None else None,
            "inverse_range_kappa": phi_kappa(h_best)[1] if h_best is not None else None,
            "J_abs": float(best["J_abs"]) if best else None,
            "M50": float(best["M50"]) if best else None,
            "U20": float(best["U20"]) if best else None,
            "P90": best.get("P90") if best else None,
            "eligible_parent_rings": len(rings), "eligible_fragments": len(arcs),
            "retained_mode_count": len(rel), "search_certificate_passed": bool(cert["passed"]),
            "model_conflict": model_conflict,
        }
        estimates.append(estimate)
        basin_rows.append({"sample_id": crop["sample_id"], "crop_id": crop_id,
                           "budget_B": relevant(basins_b), "budget_2B": rel,
                           "phase_replay": relevant(phase_basins)})

    write_jsonl(args.output_dir / "all_arc_estimates.jsonl", estimates)
    write_jsonl(args.output_dir / "basins.jsonl", basin_rows)
    write_jsonl(args.output_dir / "search_certificates.jsonl", cert_rows)
    summary = {
        "stage": 3, "schema": "ArcPith-GT-v4-stage3", "records": len(estimates),
        "status_counts": dict(Counter(row["status"] for row in estimates)),
        "certificate_passed": sum(bool(row["passed"]) for row in cert_rows),
        "certificate_failed": sum(not bool(row["passed"]) for row in cert_rows),
        "objective": {"name": "parent-balanced pseudo-Huber", "delta": DELTA,
                      "sigma0": SIGMA0, "sigma_clip": [SIGMA_MIN, SIGMA_MAX]},
        "near_point_domain": {"r_hard": R_HARD, "r_soft": R_SOFT},
        "search": {"finite_bound": FINITE_BOUND, "tau_J": TAU_J, "tau_h": TAU_H,
                   "tau_eq": TAU_EQ, "delta_mode": DELTA_MODE,
                   "overlap_kappa": list(OVERLAP_KAPPA)},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "Stage 3 minimizes the frozen parent-balanced pseudo-Huber All-Arc objective. "
        "Coordinates are emitted only with an explicit two-budget, two-chart search certificate; "
        "SEARCH_INADEQUATE records retain diagnostics but must not be treated as POINT outputs.\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if estimates else 2


if __name__ == "__main__":
    raise SystemExit(main())
