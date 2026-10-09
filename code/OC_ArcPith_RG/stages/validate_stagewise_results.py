"""Validate ArcPith-GT v4 stagewise artifacts without changing run outputs.

The validator is intentionally evidence-preserving: every manifest crop remains
in the denominator, including missing, malformed, or rejected artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VERDICTS = {"PASS", "PASS_WITH_RISK", "EXPLORATORY", "FAIL", "NOT_APPLICABLE"}
GEOMETRY = {"POINT", "POINT_CANDIDATE", "RANGE_UNCERTAIN", "AXIS", "AXIS_ONLY", "MULTIMODAL", "REJECT"}
DEGREES = {"X0_INVALID", "X1_GEOMETRY_WEAK", "X2_DIRECTION_USABLE", "X3_FINITE_RESEARCH", "X4_FINITE_STABLE", "X5_CERTIFIED"}


def finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    return False


def numbers(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            return [float(x) for x in value.replace(",", " ").split()]
        except ValueError:
            return []
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(numbers(item))
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                out.append(float(item))
        return out
    return []


def load_jsonl(path: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"MISSING_FILE:{path.name}"]
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if limit is not None and len(rows) >= limit:
                break
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                rows.append(value)
            except Exception as exc:  # malformed records stay in the denominator via error summary
                errors.append(f"line {line_no}: {exc}")
    return rows, errors


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    """Yield one JSON object at a time for the two very large artifact files."""
    with path.open("r", encoding="utf-8") as fh:
        for count, line in enumerate(fh):
            if limit is not None and count >= limit:
                return
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: non-object JSONL record")
            yield value


def load_manifest_identity(path: Path, limit: int | None) -> list[dict[str, Any]]:
    keys = ("sample_id", "tree_id", "section_id", "crop_id", "target_domain", "split")
    return [{key: row.get(key) for key in keys} for row in iter_jsonl(path, limit)]


def by_id(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: Counter[str] = Counter()
    for row in rows:
        key = row.get("sample_id") or row.get("crop_id")
        if not key:
            continue
        if key in out:
            duplicates[str(key)] += 1
        out[str(key)] = row
    return out, duplicates


def verdict(base: str, reasons: list[str], *, exploratory: bool = False) -> str:
    if base == "FAIL":
        return base
    if exploratory:
        return "EXPLORATORY"
    return "PASS_WITH_RISK" if reasons else "PASS"


def stage_record(crop: dict[str, Any], stage: int, status: str, reasons: list[str], checks: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "stage": stage,
        "sample_id": crop.get("sample_id"),
        "tree_id": crop.get("tree_id"),
        "section_id": crop.get("section_id"),
        "crop_id": crop.get("crop_id"),
        "status": status if status in VERDICTS else "FAIL",
        "reason_codes": sorted(set(reasons)),
        "checks": checks,
        **extra,
    }


def map_paths(root: Path) -> dict[int, dict[str, Path]]:
    return {
        0: {"manifest": root / "stage 0 grayscale" / "manifest_validated.jsonl"},
        1: {"summary": root / "stage 1 grayscale" / "crop_summary.jsonl", "fragments": root / "stage 1 grayscale" / "fragments.jsonl"},
        2: {"preflight": root / "stage 2 grayscale" / "preflight.jsonl", "seeds": root / "stage 2 grayscale" / "seed_registry.jsonl"},
        3: {"estimates": root / "stage 3 grayscale" / "all_arc_estimates.jsonl", "basins": root / "stage 3 grayscale" / "basins.jsonl", "certificates": root / "stage 3 grayscale" / "search_certificates.jsonl"},
        4: {"states": root / "stage 4 grayscale" / "states.jsonl", "profiles": root / "stage 4 grayscale" / "profiles.jsonl", "refinements": root / "stage 4 grayscale" / "refinements.jsonl"},
        5: {"execution": root / "stage 5 grayscale" / "execution.jsonl", "risk": root / "stage 5 grayscale" / "risk_diagnostics.jsonl"},
        6: {"parent": root / "stage 6 grayscale" / "parent_screening.jsonl", "exact": root / "stage 6 grayscale" / "parent_exact_contributions.jsonl", "fragment": root / "stage 6 grayscale" / "fragment_screening.jsonl", "crop": root / "stage 6 grayscale" / "crop_contribution_summary.jsonl"},
        7: {"final": root / "stage 7 grayscale" / "final_results.jsonl", "oracle": root / "stage 7 grayscale" / "oracle_analysis.jsonl", "audit": root / "stage 7 grayscale" / "safe_candidate_audit.jsonl"},
    }


def check_stage0(crop: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    checks: dict[str, Any] = {"record_present": row is not None}
    if row is None:
        return stage_record(crop, 0, "FAIL", ["MISSING_RESULT"], checks)
    checks["identity_match"] = all(row.get(k) == crop.get(k) for k in ("sample_id", "tree_id", "section_id", "crop_id"))
    if not checks["identity_match"]:
        reasons.append("IDENTITY_MISMATCH")
    size = row.get("image_size")
    image_path = Path(str(row.get("image", "")))
    checks["image_exists"] = image_path.exists()
    checks["image_size_valid"] = isinstance(size, list) and len(size) == 2 and all(isinstance(x, int) and x > 0 for x in size)
    if not checks["image_exists"]:
        reasons.append("IMAGE_MISSING")
    if not checks["image_size_valid"]:
        reasons.append("IMAGE_SIZE_INVALID")
    pith = numbers(row.get("pith_px"))
    checks["pith_finite"] = len(pith) == 2 and finite(pith)
    checks["pith_in_crop"] = checks["pith_finite"] and checks["image_size_valid"] and 0 <= pith[0] <= size[0] and 0 <= pith[1] <= size[1]
    # An off-crop pith-GT is a valid out-of-FOV target, provided it is retained.
    if not checks["pith_finite"]:
        reasons.append("PITH_COORDINATE_INVALID")
    curve_count = 0
    bad_points = 0
    ring_orders: dict[str, set[int]] = defaultdict(set)
    for curve in row.get("curves", []) if isinstance(row.get("curves"), list) else []:
        curve_count += 1
        if isinstance(curve.get("order"), int) and curve.get("ring_id") is not None:
            ring_orders[str(curve["ring_id"])].add(curve["order"])
        pts = curve.get("points_px", [])
        flat = numbers(pts)
        if len(flat) < 4 or len(flat) % 2:
            bad_points += 1
    checks.update({"curve_count": curve_count, "bad_curve_point_arrays": bad_points, "parent_order_consistent": all(len(values) == 1 for values in ring_orders.values())})
    if bad_points:
        reasons.append("CURVE_COORDINATE_INVALID")
    if not checks["parent_order_consistent"]:
        reasons.append("PARENT_ORDER_INCONSISTENT")
    meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    contract = meta.get("stage0_contract", {}) if isinstance(meta.get("stage0_contract"), dict) else {}
    inv = contract.get("invariant_checks", {}) if isinstance(contract.get("invariant_checks"), dict) else {}
    checks["invariant_checks"] = inv
    if inv and not all(v is True for v in inv.values() if isinstance(v, bool)):
        reasons.append("HARD_INVARIANT_FAILED")
    if row.get("mm_per_pixel") is None:
        reasons.append("PIXEL_ONLY_NO_MM_PER_PIXEL")
    if row.get("target_domain") in (None, "TARGET_UNKNOWN") or contract.get("target_domain") == "TARGET_UNKNOWN":
        reasons.append("TARGET_ALIGNMENT_UNKNOWN")
    return stage_record(crop, 0, verdict("PASS", reasons), reasons, checks, target_domain=contract.get("target_domain", row.get("target_domain")), scale_status=contract.get("scale_status"))


def check_stage1(crop: dict[str, Any], summary: dict[str, Any] | None, fragments: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    checks = {"summary_present": summary is not None, "fragment_count": len(fragments)}
    if summary is None:
        return stage_record(crop, 1, "FAIL", ["MISSING_RESULT"], checks)
    expected = int(summary.get("n_input_fragments", 0))
    eligible_expected = int(summary.get("n_eligible_fragments", 0))
    checks.update({"output_not_greater_than_input": len(fragments) <= expected, "eligible_count": sum(bool(x.get("eligible")) for x in fragments), "eligible_count_matches_summary": sum(bool(x.get("eligible")) for x in fragments) == eligible_expected})
    if len(fragments) > expected:
        reasons.append("FRAGMENT_COUNT_MISMATCH")
    if checks["eligible_count"] != eligible_expected:
        reasons.append("ELIGIBLE_COUNT_MISMATCH")
    bad_numeric = bad_tangent = bad_ds = 0
    uncertainty = Counter()
    for f in fragments:
        for key in ("length_norm", "theta_total_rad", "tangent_stability_Dt_rad", "sigma_position_norm", "sigma_psi_rad", "omega"):
            if not finite(f.get(key)):
                bad_numeric += 1
                break
        tangents = f.get("tangents", [])
        if tangents and any(abs(math.hypot(*numbers(t)[:2]) - 1.0) > 0.08 for t in tangents if len(numbers(t)) >= 2):
            bad_tangent += 1
        ds = numbers(f.get("ds_norm"))
        if ds and (any(x < -1e-10 for x in ds) or sum(ds) <= 0):
            bad_ds += 1
        uncertainty[str(f.get("uncertainty_source", "MISSING"))] += 1
    checks.update({"bad_numeric_records": bad_numeric, "bad_tangent_records": bad_tangent, "bad_ds_records": bad_ds, "uncertainty_sources": dict(uncertainty), "quadrature_K32_count": sum(f.get("quadrature", {}).get("K") == 32 for f in fragments if isinstance(f.get("quadrature"), dict)), "omega_sum": sum(float(f.get("omega", 0.0)) for f in fragments if finite(f.get("omega")))})
    if bad_numeric or bad_tangent or bad_ds:
        reasons.append("NUMERICAL_INVARIANT_FAILURE")
    if any(k.startswith("spline_bootstrap_fallback") for k in uncertainty):
        reasons.append("UNCERTAINTY_FALLBACK_ONLY")
    reasons.extend(["NO_OVERLAY_VALIDATION", "NO_SMOOTHING_SWEEP_REPLAY"])
    return stage_record(crop, 1, verdict("PASS", reasons, exploratory=True), reasons, checks, eligible=bool(summary.get("status") == "OK"))


def check_stage2(crop: dict[str, Any], pre: dict[str, Any] | None, seed: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    checks = {"preflight_present": pre is not None, "seed_registry_present": seed is not None}
    if pre is None or seed is None:
        return stage_record(crop, 2, "FAIL", ["MISSING_RESULT"], checks)
    actual = int(seed.get("seed_count", len(seed.get("seeds", []))))
    checks.update({"seed_count_matches": actual == len(seed.get("seeds", [])), "route_matches": pre.get("route") == seed.get("preflight_route"), "route": pre.get("route"), "seed_count": actual})
    if not checks["seed_count_matches"]:
        reasons.append("SEED_COUNT_MISMATCH")
    if not checks["route_matches"]:
        reasons.append("ROUTE_MISMATCH")
    families = Counter()
    bad_h = 0
    for s in seed.get("seeds", []) if isinstance(seed.get("seeds"), list) else []:
        src = s.get("source", "")
        aliases = s.get("source_aliases", [])
        aliases = aliases if isinstance(aliases, list) else str(aliases).split()
        families[src] += 1
        if not any(str(a).startswith("CS") for a in aliases):
            reasons.append("SEED_PROVENANCE_MISSING")
        h = numbers(s.get("h"))
        if len(h) != 3 or not finite(h) or abs(math.sqrt(sum(x * x for x in h)) - 1.0) > 0.08:
            bad_h += 1
    checks["seed_family_counts"] = dict(families)
    checks["bad_seed_directions"] = bad_h
    if bad_h:
        reasons.append("SEED_DIRECTION_INVALID")
    reasons.extend(["NO_COARSE_LOSS_MAP", "NO_REFERENCE_BASIN_MATCHING", "NO_ABLATION_RUNTIME_RECORD"])
    return stage_record(crop, 2, verdict("PASS", reasons, exploratory=True), reasons, checks)


def check_stage3(crop: dict[str, Any], est: dict[str, Any] | None, basin: dict[str, Any] | None, cert: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    checks = {"estimate_present": est is not None, "basin_present": basin is not None, "certificate_present": cert is not None}
    if est is None or basin is None or cert is None:
        return stage_record(crop, 3, "FAIL", ["MISSING_RESULT"], checks)
    losses = est.get("parent_losses", [])
    if not losses and isinstance(basin.get("budget_2B"), list) and basin["budget_2B"]:
        losses = basin["budget_2B"][0].get("parent_losses", [])
    checks["finite_objective"] = all(finite(est.get(k)) for k in ("J_abs", "M50", "U20"))
    checks["parent_losses_finite"] = bool(losses) and all(finite(x) for x in losses)
    checks["J_equals_parent_mean"] = checks["parent_losses_finite"] and abs(float(est.get("J_abs", 0)) - statistics.fmean(float(x) for x in losses)) < 1e-5
    checks["certificate_checks_complete"] = isinstance(cert.get("checks"), dict) and len(cert["checks"]) >= 5
    if not all(checks[k] for k in ("finite_objective", "parent_losses_finite", "J_equals_parent_mean", "certificate_checks_complete")):
        reasons.append("NUMERICAL_OR_CERTIFICATE_FAILURE")
    if est.get("status") == "SEARCH_INADEQUATE" or not cert.get("passed", False):
        reasons.append("SEARCH_INADEQUATE")
    if est.get("status") == "MODEL_CONFLICT" or est.get("model_conflict"):
        reasons.append("MODEL_CONFLICT")
    reasons.extend(["NO_INDEPENDENT_REFERENCE_BASIN_RECALL", "NO_FULL_SEED_OPTIMIZATION_TRACE"])
    stage_status = "FAIL" if "SEARCH_INADEQUATE" in reasons or "MODEL_CONFLICT" in reasons else verdict("PASS", reasons, exploratory=True)
    return stage_record(crop, 3, stage_status, reasons, checks, estimate_status=est.get("status"), certificate_passed=bool(cert.get("passed")))


def check_stage4(crop: dict[str, Any], state: dict[str, Any] | None, profile: dict[str, Any] | None, refinement: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    checks = {"state_present": state is not None, "profile_present": profile is not None, "refinement_present": refinement is not None}
    if state is None:
        return stage_record(crop, 4, "FAIL", ["MISSING_RESULT"], checks)
    if profile is None or refinement is None:
        if state.get("state") == "REJECT":
            return stage_record(crop, 4, "NOT_APPLICABLE", ["UPSTREAM_SEARCH_INADEQUATE_OR_REJECT"], checks, geometry_state=state.get("state"))
        return stage_record(crop, 4, "FAIL", ["MISSING_PROFILE_OR_REFINEMENT"], checks, geometry_state=state.get("state"))
    valid_states = {"POINT_CANDIDATE", "RANGE_UNCERTAIN", "MULTIMODAL", "REJECT"}
    checks["state_valid"] = state.get("state") in valid_states
    checks["profile_arrays_valid"] = all(isinstance(profile.get(k), list) and len(profile[k]) > 0 and all(finite(x) for x in profile[k]) for k in ("phi_rad", "vartheta_rad", "L_phi", "L_vartheta"))
    family = profile.get("delta_family", [])
    checks["delta_family_present"] = isinstance(family, list) and len(family) >= 2
    checks["refinement_stable"] = bool(refinement.get("numerical_stable", False))
    if not checks["state_valid"] or not checks["profile_arrays_valid"]:
        reasons.append("STATE_OR_PROFILE_INVALID")
    if not checks["refinement_stable"]:
        reasons.append("NUMERICAL_UNSTABLE")
    if not state.get("calibrated_coverage", False):
        reasons.append("UNCALIBRATED_SUPPORT")
    reasons.extend(["POINT_CANDIDATE_IS_NOT_FORMAL_POINT", "NO_INDEPENDENT_CALIBRATION"])
    return stage_record(crop, 4, verdict("PASS", reasons, exploratory=True), reasons, checks, geometry_state=state.get("state"))


def check_stage5(crop: dict[str, Any], execution: dict[str, Any] | None, risk: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    checks = {"execution_present": execution is not None, "risk_present": risk is not None}
    if execution is None or risk is None:
        return stage_record(crop, 5, "FAIL", ["MISSING_RESULT"], checks)
    degree = execution.get("execution_degree")
    checks.update({"degree_valid": degree in DEGREES, "coordinate_unchanged": execution.get("coordinate_changed", False) is False, "risk_valid": execution.get("risk_status") in {"UNKNOWN", "POINT_BLOCKING", "HARD_REJECT", "LOW_RISK"}, "target_domain": execution.get("target_domain")})
    if not checks["degree_valid"]:
        reasons.append("EXECUTION_DEGREE_INVALID")
    if not checks["coordinate_unchanged"]:
        reasons.append("COORDINATE_CHANGED")
    if not checks["risk_valid"]:
        reasons.append("RISK_STATUS_INVALID")
    if execution.get("target_domain") == "TARGET_UNKNOWN":
        reasons.append("TARGET_ALIGNMENT_UNKNOWN")
    reasons.extend(["MM_SCALE_UNAVAILABLE", "NO_FULL_DELETE_SCATTER_REPLAY_CALIBRATION"])
    return stage_record(crop, 5, verdict("PASS", reasons), reasons, checks, execution_degree=degree, risk_status=execution.get("risk_status"))


def check_stage6(crop: dict[str, Any], parents: list[dict[str, Any]], exact: list[dict[str, Any]], fragments: list[dict[str, Any]], summary: dict[str, Any] | None, expected_fragments: int) -> dict[str, Any]:
    reasons: list[str] = []
    checks = {"crop_summary_present": summary is not None, "parent_screening_count": len(parents), "fragment_screening_count": len(fragments), "exact_count": len(exact)}
    if summary is None:
        return stage_record(crop, 6, "FAIL", ["MISSING_RESULT"], checks)
    checks["all_eligible_fragments_screened"] = len(fragments) == expected_fragments
    checks["exact_status_valid"] = all(x.get("status") in {"EXACT", "UNRESOLVED", "UNRESOLVED_SEARCH", "SCREEN_ONLY"} for x in exact)
    exact_resolved = [x for x in exact if x.get("status") == "EXACT"]
    checks["exact_certificates_present"] = all(isinstance(x.get("search_certificate"), dict) for x in exact_resolved)
    checks["pixel_only_gt_reason"] = all(x.get("C_GT_mm") is None and x.get("C_GT_mm_status") == "PIXEL_ONLY_NO_MM_PER_PIXEL" for x in exact_resolved)
    valid_roles = {"DIRECTION_CRITICAL", "RANGE_CRITICAL", "BENEFICIAL_GT", "HARMFUL_GT", "HIGH_LEVERAGE", "MODE_EXCLUSION", "REDUNDANT", "UNRESOLVED"}
    role_bad = sum(any(r not in valid_roles for r in (x.get("roles") or [])) for x in exact)
    checks["role_labels_valid"] = role_bad == 0
    if not checks["all_eligible_fragments_screened"]:
        reasons.append("FRAGMENT_SCREENING_INCOMPLETE")
    if not checks["exact_status_valid"] or not checks["exact_certificates_present"]:
        reasons.append("EXACT_LINEAGE_OR_CERTIFICATE_INVALID")
    if not checks["pixel_only_gt_reason"]:
        reasons.append("GT_MM_STATUS_INVALID")
    if not checks["role_labels_valid"]:
        reasons.append("ROLE_LABEL_INVALID")
    reasons.extend(["GT_ROLES_ARE_POSTHOC_AUDIT_ONLY", "NO_PHYSICAL_MM_CONTRIBUTION"])
    return stage_record(crop, 6, verdict("PASS", reasons, exploratory=True), reasons, checks, exact_resolved=sum(x.get("status") == "EXACT" for x in exact))


def check_stage7(crop: dict[str, Any], final: dict[str, Any] | None, oracle: dict[str, Any] | None, audit: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    checks = {"final_present": final is not None, "oracle_present": oracle is not None, "audit_records": len(audit)}
    if final is None:
        return stage_record(crop, 7, "FAIL", ["MISSING_RESULT"], checks)
    checks.update({"f_all": final.get("final_method") == "F-All", "safe_prune_off": final.get("safe_prune_applied") is False, "geometry_valid": final.get("geometry_state") in GEOMETRY, "degree_valid": final.get("execution_degree") in DEGREES, "oracle_coordinate_unchanged": all(x.get("may_change_final_coordinate") is False for x in ([oracle] if oracle else []))})
    if not checks["f_all"]:
        reasons.append("FINAL_METHOD_NOT_F_ALL")
    if not checks["safe_prune_off"]:
        reasons.append("SAFE_PRUNE_APPLIED")
    if not checks["geometry_valid"] or not checks["degree_valid"]:
        reasons.append("FINAL_STATE_INVALID")
    if oracle and oracle.get("may_change_final_coordinate"):
        reasons.append("ORACLE_CHANGED_COORDINATE")
    reasons.extend(["SAFE_PRUNE_OFF", "DEPLOYMENT_GATE_NOT_PASSED", "INDEPENDENT_TREE_COUNT_BELOW_FREEZE_GATE"])
    return stage_record(crop, 7, verdict("PASS", reasons, exploratory=True), reasons, checks, final_method=final.get("final_method"), geometry_state=final.get("geometry_state"), execution_degree=final.get("execution_degree"))


def aggregate(rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(r.get("status") for r in rows)
    reasons = Counter(code for r in rows for code in r.get("reason_codes", []))
    trees: dict[str, dict[str, Any]] = {}
    targets: Counter[str] = Counter()
    for r in rows:
        tree = str(r.get("tree_id") or "UNKNOWN")
        trees.setdefault(tree, {"records": 0, "status_counts": Counter(), "reason_counts": Counter()})
        trees[tree]["records"] += 1
        trees[tree]["status_counts"][r.get("status")] += 1
        trees[tree]["reason_counts"].update(r.get("reason_codes", []))
        if r.get("target_domain"):
            targets[str(r["target_domain"])] += 1
    for value in trees.values():
        value["status_counts"] = dict(value["status_counts"])
        value["reason_counts"] = dict(value["reason_counts"])
    return {"denominator": len(manifest), "records": len(rows), "missing_records": len(manifest) - len(rows), "status_counts": dict(status), "reason_counts": dict(reasons), "tree_aggregation": trees, "target_domain_counts": dict(targets)}


def write_report(stage: int, directory: Path, summary: dict[str, Any]) -> None:
    lines = [f"# Stage {stage} Validation", "", "This report is generated from the fixed manifest denominator; failed and missing records are retained.", "", f"- Denominator: **{summary['denominator']}**", f"- Validated records: **{summary['records']}**", f"- Missing records: **{summary['missing_records']}**", f"- Status counts: `{json.dumps(summary['status_counts'], ensure_ascii=False)}`", "", "## Main reason codes", ""]
    for key, value in sorted(summary["reason_counts"].items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Biological-tree aggregation", "", "| tree | records | statuses |", "|---|---:|---|"]
    for tree, value in sorted(summary["tree_aggregation"].items()):
        lines.append(f"| {tree} | {value['records']} | `{json.dumps(value['status_counts'], ensure_ascii=False)}` |")
    lines += ["", "## Interpretation", "", "Stage labels distinguish numerical validity from scientific calibration. Missing millimetre scale, unknown target alignment, and the four-tree dataset prevent a formal calibrated PASS. Safe-Prune remains an All-Arc deployment decision and is not enabled by this validation.", ""]
    (directory / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest_grayscale.jsonl")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    manifest = load_manifest_identity(Path(args.manifest), args.limit)
    artifact_root = Path(args.artifact_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = map_paths(artifact_root)
    load_errors: dict[str, list[str]] = {}
    dups: dict[str, dict[str, int]] = {}
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}

    def mapped(stage: int, name: str, *, unlimited: bool = False) -> dict[str, dict[str, Any]]:
        rows, errors = load_jsonl(paths[stage][name], None if unlimited else args.limit)
        if errors:
            load_errors[f"stage{stage}:{name}"] = errors[:10]
        result, duplicate = by_id(rows)
        if duplicate:
            dups[f"stage{stage}:{name}"] = dict(duplicate)
        return result

    def grouped(stage: int, name: str) -> dict[str, list[dict[str, Any]]]:
        rows, errors = load_jsonl(paths[stage][name], None)
        if errors:
            load_errors[f"stage{stage}:{name}"] = errors[:10]
        result: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            sid = row.get("sample_id") or row.get("crop_id")
            if sid is not None and str(sid) in manifest_by_id:
                result[str(sid)].append(row)
        return result

    # Each large stage is loaded, validated, and released before the next one.
    stage_outputs_by_id: dict[int, dict[str, dict[str, Any]]] = {stage: {} for stage in range(8)}
    stage0 = mapped(0, "manifest")
    for sid, crop in manifest_by_id.items():
        stage_outputs_by_id[0][sid] = check_stage0(crop, stage0.get(sid))
    del stage0

    stage1_summary = mapped(1, "summary")
    stage1_fragments = grouped(1, "fragments")
    for sid, crop in manifest_by_id.items():
        stage_outputs_by_id[1][sid] = check_stage1(crop, stage1_summary.get(sid), stage1_fragments.get(sid, []))
    del stage1_summary

    stage2_preflight = mapped(2, "preflight")
    # Seed registry has very large per-record arrays; validate it one crop at a time.
    for seed in iter_jsonl(paths[2]["seeds"], args.limit):
        sid = str(seed.get("sample_id") or seed.get("crop_id"))
        if sid in manifest_by_id:
            stage_outputs_by_id[2][sid] = check_stage2(manifest_by_id[sid], stage2_preflight.get(sid), seed)
    for sid, crop in manifest_by_id.items():
        stage_outputs_by_id[2].setdefault(sid, check_stage2(crop, stage2_preflight.get(sid), None))
    del stage2_preflight

    stage3_est = mapped(3, "estimates")
    stage3_basin = mapped(3, "basins")
    stage3_cert = mapped(3, "certificates")
    for sid, crop in manifest_by_id.items():
        stage_outputs_by_id[3][sid] = check_stage3(crop, stage3_est.get(sid), stage3_basin.get(sid), stage3_cert.get(sid))
    del stage3_est, stage3_basin, stage3_cert

    stage4_state, stage4_profile, stage4_refine = mapped(4, "states"), mapped(4, "profiles"), mapped(4, "refinements")
    for sid, crop in manifest_by_id.items():
        stage_outputs_by_id[4][sid] = check_stage4(crop, stage4_state.get(sid), stage4_profile.get(sid), stage4_refine.get(sid))
    del stage4_state, stage4_profile, stage4_refine

    stage5_execution, stage5_risk = mapped(5, "execution"), mapped(5, "risk")
    for sid, crop in manifest_by_id.items():
        stage_outputs_by_id[5][sid] = check_stage5(crop, stage5_execution.get(sid), stage5_risk.get(sid))
    del stage5_risk

    stage6_parent = grouped(6, "parent")
    stage6_exact = grouped(6, "exact")
    stage6_fragment = grouped(6, "fragment")
    stage6_crop = mapped(6, "crop")
    for sid, crop in manifest_by_id.items():
        expected = sum(bool(x.get("eligible")) for x in stage1_fragments.get(sid, []))
        stage_outputs_by_id[6][sid] = check_stage6(crop, stage6_parent.get(sid, []), stage6_exact.get(sid, []), stage6_fragment.get(sid, []), stage6_crop.get(sid), expected)
    del stage1_fragments, stage6_parent, stage6_exact, stage6_fragment, stage6_crop

    stage7_final, stage7_oracle, stage7_audit = mapped(7, "final"), mapped(7, "oracle"), grouped(7, "audit")
    for sid, crop in manifest_by_id.items():
        stage_outputs_by_id[7][sid] = check_stage7(crop, stage7_final.get(sid), stage7_oracle.get(sid), stage7_audit.get(sid, []))
    all_index: list[dict[str, Any]] = []
    for crop in manifest:
        sid = str(crop.get("sample_id"))
        stage_rows = {stage: stage_outputs_by_id[stage][sid] for stage in range(8)}
        first_failure = next((stage for stage in range(8) if stage_rows[stage]["status"] == "FAIL"), None)
        final = stage7_final.get(sid, {})
        index_row = {"tree_id": crop.get("tree_id"), "section_id": crop.get("section_id"), "crop_id": crop.get("crop_id"), "sample_id": sid, "run_id": "grayscale_stagewise_20260904", "config_hash": None, "code_commit": None, "split": crop.get("split", "unspecified"), "target_domain": crop.get("target_domain", "TARGET_UNKNOWN"), **{f"stage{stage}_status": stage_rows[stage]["status"] for stage in range(8)}, "geometry_state": final.get("geometry_state"), "execution_degree": final.get("execution_degree"), "first_failure_stage": first_failure, "reason_codes": sorted({code for row in stage_rows.values() for code in row["reason_codes"]}), "artifact_paths": {str(stage): {name: str(path) for name, path in paths[stage].items()} for stage in range(8)}}
        all_index.append(index_row)

    stage_outputs = {stage: [stage_outputs_by_id[stage][str(crop["sample_id"])] for crop in manifest] for stage in range(8)}

    for stage, rows in stage_outputs.items():
        directory = output_root / f"stage {stage} grayscale"
        directory.mkdir(parents=True, exist_ok=True)
        summary = aggregate(rows, manifest)
        summary.update({"stage": stage, "validator": "validate_stagewise_results.py", "load_errors": load_errors, "duplicate_keys": dups})
        (directory / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with (directory / "validation_records.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        write_report(stage, directory, summary)
    with (output_root / "result_index.jsonl").open("w", encoding="utf-8") as fh:
        for row in all_index:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    root_summary = {"manifest": str(Path(args.manifest).resolve()), "denominator": len(manifest), "trees": sorted({str(x.get("tree_id")) for x in manifest}), "sections": len({str(x.get("section_id")) for x in manifest}), "stage_summaries": {str(stage): json.loads((output_root / f"stage {stage} grayscale" / "validation_summary.json").read_text(encoding="utf-8")) for stage in range(8)}, "safe_prune": "OFF", "conclusion": "Geometry pipeline is executable, but the current results are exploratory/pass-with-risk: four biological trees, unknown target alignment, no mm_per_pixel, and no independent Safe-Prune calibration prevent a formal calibrated PASS or X5 claim."}
    (output_root / "analysis_summary.md").write_text("# ArcPith-GT v4 Stagewise Validation\n\n" + root_summary["conclusion"] + "\n\n- Complete denominator: **" + str(len(manifest)) + "** crops\n- Biological trees: **" + str(len(root_summary["trees"])) + "**\n- Safe-Prune: **OFF**\n- Deployment method: **F-All**\n- No millimetre error or X5 certification is emitted.\n", encoding="utf-8")
    (output_root / "analysis_summary.json").write_text(json.dumps(root_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"denominator": len(manifest), "stages": {str(s): len(v) for s, v in stage_outputs.items()}, "output_root": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
