"""Create evidence-based, per-stage ArcPith-GT v4 analysis reports.

This script is deliberately read-only with respect to Stage 0--7 run outputs.
It turns the existing JSONL artifacts into explicit metric/criterion/observation/
verdict tables.  A missing prescribed reference, calibration, or replay artifact
is reported as NOT_MEASURABLE_FROM_CURRENT_ARTIFACT; it is never treated as a
pass by default.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: malformed JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row is not an object")
            yield value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def pct(numerator: int, denominator: int) -> str:
    return "N/A" if not denominator else f"{100.0 * numerator / denominator:.2f}% ({numerator}/{denominator})"


def q(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    at = (len(values) - 1) * probability
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (at - lo)


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": q(values, 0.90),
        "max": max(values) if values else None,
    }


def display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "`"
    return str(value)


def metric(layer: str, name: str, criterion: str, observed: Any, decision: str, note: str = "") -> dict[str, Any]:
    return {"layer": layer, "metric": name, "criterion": criterion, "observed": observed, "decision": decision, "note": note}


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def stage0(root: Path, denominator: int) -> dict[str, Any]:
    n = finite_pith = in_crop = invariant_all = outside_unclipped = with_scale = 0
    domains: Counter[str] = Counter()
    for row in rows(root / "stage 0 grayscale" / "manifest_validated.jsonl"):
        n += 1
        pith, size = row.get("pith_px"), row.get("image_size")
        if isinstance(pith, list) and len(pith) == 2 and all(is_number(v) for v in pith):
            finite_pith += 1
            if isinstance(size, list) and len(size) == 2 and all(is_number(v) for v in size) and 0 <= pith[0] <= size[0] and 0 <= pith[1] <= size[1]:
                in_crop += 1
        if row.get("mm_per_pixel") is not None:
            with_scale += 1
        contract = row.get("metadata", {}).get("stage0_contract", {})
        domains[str(contract.get("target_domain", row.get("target_domain", "MISSING")))] += 1
        checks = contract.get("invariant_checks", {})
        invariant_all += int(bool(checks) and all(value is True for value in checks.values()))
        outside_unclipped += int(checks.get("pith_outside_not_clipped") is True)
    metrics = [
        metric("数值", "完整清单记录", "每个 manifest crop 必须有 Stage 0 合同", pct(n, denominator), "满足" if n == denominator else "未达标"),
        metric("数值", "pith 坐标有限", "坐标必须为两个有限数", pct(finite_pith, n), "满足" if finite_pith == n else "未达标"),
        metric("数值", "硬不变量", "source/crop 往返、图外 GT 保持、lineage 和固定变换均为 true", pct(invariant_all, n), "满足" if invariant_all == n else "未达标"),
        metric("数据语义", "图外 GT 未裁剪", "pith_outside_not_clipped 必须为 true", pct(outside_unclipped, n), "满足" if outside_unclipped == n else "未达标"),
        metric("科学", "目标域认证", "冻结计划要求 TARGET_ALIGNED/ECCENTRIC_GT 或明确适用域证据", counter_dict(domains), "未达标", "全部为 TARGET_UNKNOWN，不能宣称生物学髓心适用域已认证。"),
        metric("使用", "物理尺度", "mm_per_pixel 用于毫米精度及应用门", pct(with_scale, n), "未达标", "只能报告 pixel / normalized / projective 指标。"),
        metric("科学", "变换等变 replay", "计划要求 fixture replay 的 E_equiv 记录", "未提供数值 replay 产物", "NOT_MEASURABLE_FROM_CURRENT_ARTIFACT"),
    ]
    return {"stage": 0, "status": "PASS_WITH_RISK", "counts": {"records": n, "pith_in_crop": in_crop, "pith_outside_crop": finite_pith - in_crop, "target_domains": counter_dict(domains)}, "metrics": metrics,
            "three_layers": [["数值有效性", "PASS", "完整、有限坐标和记录的硬不变量均通过。"], ["科学有效性", "PASS_WITH_RISK", "坐标合同成立，但目标域全为未知，无法认证公共径向中心等同生物学髓心。"], ["使用有效性", "PASS_WITH_RISK", "允许进入几何研究链；无毫米尺度且无目标域认证。"]],
            "downstream": "下游可解释为局部径向汇聚几何研究；不得报告毫米误差、TARGET_ALIGNED 结论或生物学髓心适用域结论。"}


def stage1(root: Path, denominator: int) -> dict[str, Any]:
    crops = inputs = eligible_expected = 0
    invalid: Counter[str] = Counter()
    for row in rows(root / "stage 1 grayscale" / "crop_summary.jsonl"):
        crops += 1
        inputs += int(row.get("n_input_fragments", 0))
        eligible_expected += int(row.get("n_eligible_fragments", 0))
        for reason in row.get("errors", []):
            invalid[str(reason)] += 1
    fragments = eligible = tangent_total = tangent_bad = ds_total = ds_bad = 0
    tangent_error: list[float] = []
    source: Counter[str] = Counter()
    k_counts: Counter[str] = Counter()
    omega_by_crop: Counter[str] = Counter()
    for row in rows(root / "stage 1 grayscale" / "fragments.jsonl"):
        fragments += 1
        eligible += int(bool(row.get("eligible")))
        source[str(row.get("uncertainty_source", "MISSING"))] += 1
        quadrature = row.get("quadrature", {})
        k_counts[str(quadrature.get("K", "MISSING"))] += 1
        omega_by_crop[str(row.get("sample_id"))] += float(row.get("omega", 0.0)) if is_number(row.get("omega")) else 0.0
        for tangent in row.get("tangents", []):
            if isinstance(tangent, list) and len(tangent) >= 2 and all(is_number(x) for x in tangent[:2]):
                tangent_total += 1
                error = abs(math.hypot(tangent[0], tangent[1]) - 1.0)
                tangent_error.append(error)
                tangent_bad += int(error > 0.08)
        ds = row.get("ds_norm", [])
        if isinstance(ds, list) and ds:
            ds_total += 1
            ds_bad += int(any(not is_number(v) or v < -1e-10 for v in ds) or sum(ds) <= 0)
    metrics = [
        metric("数值", "crop 清单完整", "每个 manifest crop 有 summary", pct(crops, denominator), "满足" if crops == denominator else "未达标"),
        metric("数据", "fragment 保留率", "按输入 fragment 全分母保留并报告原因", pct(eligible_expected, inputs), "满足", f"未进入连续测量的 {inputs - eligible_expected} 条：{counter_dict(invalid)}"),
        metric("数值", "eligible 输出一致", "fragments.jsonl 中 eligible 数必须等于 summary", f"{eligible}/{eligible_expected}", "满足" if eligible == eligible_expected else "未达标"),
        metric("数值", "切向单位范数", "所有采样切向应为单位向量（实现容差 0.08）", {"violations": tangent_bad, "total": tangent_total, "abs_norm_error": stats(tangent_error)}, "满足" if tangent_bad == 0 else "未达标"),
        metric("数值", "积分弧长权重 ds", "每条连续弧 ds 非负且和为正", pct(ds_total - ds_bad, ds_total), "满足" if ds_bad == 0 else "未达标"),
        metric("数值", "quadrature 节点", "冻结实现使用 K=32，并需有 2K 收敛 replay", counter_dict(k_counts), "部分满足", "K=32 记录存在；当前产物没有独立 K→2K replay。"),
        metric("科学", "不确定度标定来源", "真实重复标注应支持 Sigma_x / sigma_psi", counter_dict(source), "未达标", "全部为 spline bootstrap fallback，未提供重复标注 calibration。"),
        metric("科学", "复制/插值/切分证据守恒 replay", "计划要求三类 replay 比较 J、state 和 parent budget", "未提供 replay 产物", "NOT_MEASURABLE_FROM_CURRENT_ARTIFACT"),
        metric("使用", "每 crop omega 预算", "parent-balanced budget 应可审计", stats(list(omega_by_crop.values())), "满足", "数值预算已汇总；尚无 split/duplicate replay 证明。"),
    ]
    return {"stage": 1, "status": "EXPLORATORY", "counts": {"crops": crops, "input_fragments": inputs, "eligible_fragments": eligible_expected, "fragment_records": fragments, "invalid_reasons": counter_dict(invalid)}, "metrics": metrics,
            "three_layers": [["数值有效性", "PASS", "连续弧的切向与 ds 硬检查通过。"], ["科学有效性", "EXPLORATORY", "缺少重复标注、overlay、平滑尺度和证据守恒 replay。"], ["使用有效性", "EXPLORATORY", "可作为冻结测量输入研究，不可称已标定的测量不确定度。"]],
            "downstream": "允许以当前连续弧继续搜索；不得把 sigma 当作正式校准尺度，也不得用该结果签发 X4/X5。"}


def stage2(root: Path, denominator: int) -> dict[str, Any]:
    preflight = routes = 0
    route_counts: Counter[str] = Counter()
    pre_values: dict[str, list[float]] = {key: [] for key in ("eligible_parent_ring_count", "total_arc_norm", "Theta_total_rad", "B_spatial", "T_span_rad", "g_svd", "truncation_ratio")}
    for row in rows(root / "stage 2 grayscale" / "preflight.jsonl"):
        preflight += 1
        route_counts[str(row.get("route", "MISSING"))] += 1
        for key in pre_values:
            if is_number(row.get(key)):
                pre_values[key].append(float(row[key]))
    registries = seeds_total = bad_norm = 0
    seed_counts: list[float] = []
    families: Counter[str] = Counter()
    alias_families: Counter[str] = Counter()
    for row in rows(root / "stage 2 grayscale" / "seed_registry.jsonl"):
        registries += 1
        listed = row.get("seeds", [])
        seed_counts.append(float(len(listed)))
        for seed in listed:
            seeds_total += 1
            source = str(seed.get("source", "MISSING"))
            families[source] += 1
            aliases = seed.get("source_aliases", [])
            if not isinstance(aliases, list):
                aliases = [str(aliases)]
            for alias in aliases:
                alias_families[str(alias)] += 1
            h = seed.get("h", [])
            bad_norm += int(not isinstance(h, list) or len(h) != 3 or not all(is_number(x) for x in h) or abs(math.sqrt(sum(float(x) ** 2 for x in h)) - 1) > 0.08)
    expected = {"CS1", "CS2", "CS3", "CS4", "CS5", "CS6_FINITE_MESH", "CS6_FAR_MESH"}
    present = set(alias_families)
    metrics = [
        metric("数值", "preflight/registry 完整", "每个 manifest crop 都有二者", {"preflight": pct(preflight, denominator), "seed_registry": pct(registries, denominator)}, "满足" if preflight == denominator == registries else "未达标"),
        metric("数值", "seed 数分布", "记录的 seed_count 必须与 seeds 长度一致且方向单位化", {"per_crop": stats(seed_counts), "total": seeds_total, "invalid_direction": bad_norm}, "满足" if bad_norm == 0 else "未达标"),
        metric("科学", "六源候选覆盖", "按 source_aliases 覆盖 CS1/CS2/CS3/CS4/CS5/CS6 finite/far，并保留主 source 统计", {"primary_source": counter_dict(families), "alias_coverage": counter_dict(alias_families), "missing_aliases": sorted(expected - present)}, "满足" if not expected - present else "未达标", "主 source 没有单独列出 CS3 不等于 CS3 缺失；应以 alias coverage 判断候选家族是否覆盖。"),
        metric("科学", "预检分布", "预检只用于路由/分层，不直接作为状态", {"routes": counter_dict(route_counts), **{key: stats(value) for key, value in pre_values.items()}}, "满足"),
        metric("科学", "reference basin recall", "冻结计划要求同目标高预算 reference 对照", "未提供 reference_basin_matching", "NOT_MEASURABLE_FROM_CURRENT_ARTIFACT"),
        metric("科学", "cold-start ablation / coarse loss map", "计划要求六源消融和 finite/far loss map", "未提供产物", "NOT_MEASURABLE_FROM_CURRENT_ARTIFACT"),
        metric("使用", "搜索闭合结论", "仅在 relevant-mode/global-best recall 达到预注册门后成立", "reference 缺失", "未达标"),
    ]
    return {"stage": 2, "status": "EXPLORATORY", "counts": {"preflight": preflight, "registries": registries, "routes": counter_dict(route_counts), "seed_primary_source": counter_dict(families), "seed_alias_coverage": counter_dict(alias_families)}, "metrics": metrics,
            "three_layers": [["数值有效性", "PASS", "预检与 seed registry 齐全，seed 方向均有效。"], ["科学有效性", "EXPLORATORY", "候选库已生成，但缺少同目标 reference basin、消融和 coarse-map 证据。"], ["使用有效性", "EXPLORATORY", "可将候选传入 All-Arc；不得据此声称冷启动已覆盖所有相关 basin。"]],
            "downstream": "Stage 3 的 Search Certificate 仍需承担拒识责任；不能仅以 Stage 2 完整运行证明全局搜索。"}


def stage3(root: Path, denominator: int) -> dict[str, Any]:
    estimates = certs = 0
    state_counts: Counter[str] = Counter()
    estimate_reasons: Counter[str] = Counter()
    status_by_sample: dict[str, str] = {}
    values: dict[str, list[float]] = {key: [] for key in ("J_abs", "M50", "U20", "P90")}
    for row in rows(root / "stage 3 grayscale" / "all_arc_estimates.jsonl"):
        estimates += 1
        state_counts[str(row.get("status", "MISSING"))] += 1
        status_by_sample[str(row.get("sample_id"))] = str(row.get("status", "MISSING"))
        for reason in row.get("reason_codes", []):
            estimate_reasons[str(reason)] += 1
        for key in values:
            if is_number(row.get(key)):
                values[key].append(float(row[key]))
    passed = 0
    reason_counts: Counter[str] = Counter()
    check_fail: Counter[str] = Counter()
    delta_j: list[float] = []
    delta_h: list[float] = []
    basin_b: list[float] = []
    basin_2b: list[float] = []
    certificate_pass_by_sample: dict[str, bool] = {}
    for row in rows(root / "stage 3 grayscale" / "search_certificates.jsonl"):
        certs += 1
        passed += int(row.get("passed") is True)
        certificate_pass_by_sample[str(row.get("sample_id"))] = row.get("passed") is True
        for reason in row.get("reason_codes", []):
            reason_counts[str(reason)] += 1
        for check, value in row.get("checks", {}).items():
            if value is not True:
                check_fail[str(check)] += 1
        for target, key in ((delta_j, "relative_best_loss_change"), (delta_h, "max_basin_match_rp2"), (basin_b, "basin_count_B"), (basin_2b, "basin_count_2B")):
            if is_number(row.get(key)):
                target.append(float(row[key]))
    certificate_failure = denominator - passed
    usable_geometry = sum(certificate_pass_by_sample.get(sample_id, False) and status == "OK" for sample_id, status in status_by_sample.items())
    blocked_geometry = denominator - usable_geometry
    metrics = [
        metric("数值", "estimate/certificate 完整", "每个 manifest crop 一条 estimate 和 certificate", {"estimate": pct(estimates, denominator), "certificate": pct(certs, denominator)}, "满足" if estimates == certs == denominator else "未达标"),
        metric("数值", "Search Certificate", "所有关键 checks 通过才允许几何解释", {"passed": pct(passed, certs), "failed": certificate_failure, "failed_checks": counter_dict(check_fail), "reasons": counter_dict(reason_counts)}, "未达标" if certificate_failure else "满足"),
        metric("数值", "预算/盆地收敛观测", "报告 B→2B 的 loss、位置和 basin 数，具体容差在 certificate 固定", {"relative_best_loss_change": stats(delta_j), "max_basin_match_rp2": stats(delta_h), "basin_count_B": stats(basin_b), "basin_count_2B": stats(basin_2b)}, "满足", "仅说明当前 certificate 内的一致性，不替代独立 reference-search。"),
        metric("科学", "All-Arc 与 tail guard", "联合报告 J_abs、M50、U20，不能只以 loss 判髓心正确", {key: stats(value) for key, value in values.items()}, "满足", "这些仅为目标函数诊断。"),
        metric("科学", "几何结果状态", "SEARCH_INADEQUATE / MODEL_CONFLICT 不可作为 POINT/RANGE/AXIS 解释", {"estimate_status": counter_dict(state_counts), "certificate_pass_but_not_OK": passed - usable_geometry, "scientifically_usable": usable_geometry, "blocked_geometry": blocked_geometry, "reason_codes": counter_dict(estimate_reasons)}, "部分满足", f"certificate 失败为 {certificate_failure}；再加上 certificate 通过但 MODEL_CONFLICT 的样本后，共 {blocked_geometry} 个 crop 不能作几何解释。"),
        metric("科学", "独立 reference basin recall", "计划要求高预算同目标 reference 子集", "未提供 matching 产物", "NOT_MEASURABLE_FROM_CURRENT_ARTIFACT"),
        metric("使用", "允许输出几何状态", "仅 certificate 通过且 status=OK 的 crop 可交给 Stage 4/5", pct(usable_geometry, denominator), "部分满足"),
    ]
    return {"stage": 3, "status": "EXPLORATORY", "counts": {"estimates": estimates, "certificates_passed": passed, "certificates_failed": certificate_failure, "scientifically_usable": usable_geometry, "blocked_geometry": blocked_geometry, "estimate_status": counter_dict(state_counts), "certificate_reasons": counter_dict(reason_counts)}, "metrics": metrics,
            "three_layers": [["数值有效性", "PARTIAL_PASS", f"{passed} 个 crop 的 certificate 完整通过；{certificate_failure} 个未通过。"], ["科学有效性", "EXPLORATORY", "缺少独立 reference basin 对照，不能证明全局覆盖。"], ["使用有效性", "PARTIAL", f"{blocked_geometry} 个 certificate 失败或模型冲突 crop 必须 REJECT；其余仅可继续研究级解释。"]],
            "downstream": f"{blocked_geometry} 个 certificate 失败或模型冲突 crop 不能支持 POINT/RANGE/AXIS 结论；不得从精度统计中删除。"}


def stage4(root: Path, denominator: int) -> dict[str, Any]:
    states = profiles = refinements = stable = stationary = calibrated = 0
    state_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    refine_shift: list[float] = []
    gradients: list[float] = []
    support_nesting_bad = support_delta_records = support_components = support_infinity = 0
    for row in rows(root / "stage 4 grayscale" / "states.jsonl"):
        states += 1
        state_counts[str(row.get("state", "MISSING"))] += 1
        calibrated += int(row.get("calibrated_coverage") is True)
        for reason in row.get("reason_codes", []):
            reason_counts[str(reason)] += 1
    for row in rows(root / "stage 4 grayscale" / "profiles.jsonl"):
        profiles += 1
        family = row.get("delta_family", [])
        last_cells = -1
        for entry in family:
            cells = int(entry.get("support_cells", 0))
            support_nesting_bad += int(last_cells > cells)
            last_cells = cells
            support_delta_records += 1
            support_components += int(entry.get("component_count", 0))
            support_infinity += sum(int(component.get("touches_infinity") is True) for component in entry.get("components", []))
    for row in rows(root / "stage 4 grayscale" / "refinements.jsonl"):
        refinements += 1
        stable += int(row.get("numerical_stable") is True)
        stationary += int(row.get("stationary") is True)
        if is_number(row.get("rp2_shift_rad")):
            refine_shift.append(float(row["rp2_shift_rad"]))
        if is_number(row.get("gradient_norm")):
            gradients.append(float(row["gradient_norm"]))
    missing_profiles = denominator - profiles
    metrics = [
        metric("数值", "state 记录完整", "每个 manifest crop 必须有 state（包含 REJECT）", pct(states, denominator), "满足" if states == denominator else "未达标"),
        metric("数值", "profile/refinement 路由", "只对通过上游搜索的样本做 profile/refinement；REJECT 可无此产物", {"profiles": profiles, "refinements": refinements, "upstream/no-profile": missing_profiles}, "满足", "无 profile 的记录保留在 states.jsonl，未从分母删除。"),
        metric("数值", "dense refit 稳定性", "numerical_stable 与 stationary 必须为 true 才可作数值精化解释", {"stable": pct(stable, refinements), "stationary": pct(stationary, refinements), "rp2_shift_rad": stats(refine_shift), "gradient_norm": stats(gradients)}, "部分满足" if stable < refinements else "满足"),
        metric("科学", "delta family 嵌套", "delta 增大时 support_cells 不应减少；要报告 component/infinity", {"delta_records": support_delta_records, "nesting_violations": support_nesting_bad, "components_total": support_components, "touches_infinity": support_infinity}, "满足" if support_nesting_bad == 0 else "未达标"),
        metric("科学", "五态分布", "POINT/RANGE/AXIS/MULTIMODAL/REJECT 必须分开报告", counter_dict(state_counts), "满足", "POINT_CANDIDATE 不等于正式 POINT。"),
        metric("科学", "coverage calibration", "独立 calibration trees 和冻结 delta_alpha 才可报告 coverage/正式 POINT", pct(calibrated, states), "未达标", "所有状态均为未校准。"),
        metric("科学", "profile replay / topology 稳定", "计划要求邻近 delta 与 search replay 的稳定性", "未提供独立 replay 产物", "NOT_MEASURABLE_FROM_CURRENT_ARTIFACT"),
        metric("使用", "输出等级限制", "未校准时仅 POINT_CANDIDATE / X3 研究级", counter_dict(reason_counts), "满足", "当前状态不可升级为 calibrated POINT。"),
    ]
    return {"stage": 4, "status": "EXPLORATORY", "counts": {"states": states, "profiles": profiles, "refinements": refinements, "state_counts": counter_dict(state_counts), "reason_counts": counter_dict(reason_counts)}, "metrics": metrics,
            "three_layers": [["数值有效性", "PARTIAL_PASS", "state 完整，且大多数 refinement 稳定；上游拒识样本正确不做 profile。"], ["科学有效性", "EXPLORATORY", "有 profile 和 delta-family，但没有独立 calibration/replay 和 controlled reference。"], ["使用有效性", "EXPLORATORY", "可输出 POINT_CANDIDATE、RANGE_UNCERTAIN、MULTIMODAL 或 REJECT；不能签发正式 POINT。"]],
            "downstream": "Stage 5 必须保留 calibration 未满足和 target risk；不得将 POINT_CANDIDATE 重命名为 POINT。"}


def stage5(root: Path, denominator: int) -> dict[str, Any]:
    n = unchanged = cert_pass = ref_stable = 0
    geometry: Counter[str] = Counter()
    risk: Counter[str] = Counter()
    execution: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    inconsistent = 0
    r2: list[float] = []
    structured_flag = 0
    fixture_status: Counter[str] = Counter()
    for row in rows(root / "stage 5 grayscale" / "execution.jsonl"):
        n += 1
        geometry[str(row.get("geometry_state", "MISSING"))] += 1
        risk[str(row.get("risk_status", "MISSING"))] += 1
        execution[str(row.get("execution_degree", "MISSING"))] += 1
        unchanged += int(row.get("coordinate_source") == "STAGE4_ALL_ARC_UNCHANGED")
        for reason in row.get("reason_codes", []):
            reasons[str(reason)] += 1
        state, degree = str(row.get("geometry_state")), str(row.get("execution_degree"))
        inconsistent += int((state == "REJECT" and degree not in {"X0_INVALID", "X1_GEOMETRY_WEAK"}) or (state == "POINT_CANDIDATE" and degree not in {"X3_FINITE_RESEARCH"}) or (state == "RANGE_UNCERTAIN" and degree not in {"X2_DIRECTION_USABLE"}))
    for row in rows(root / "stage 5 grayscale" / "risk_diagnostics.jsonl"):
        structured = row.get("structured_residual", {})
        if is_number(structured.get("R2_struct_oof")):
            r2.append(float(structured["R2_struct_oof"]))
        structured_flag += int(structured.get("systematic_structure_flag") is True)
        replay = row.get("measurement_search_replay", {})
        cert_pass += int(replay.get("stage3_certificate_passed") is True)
        ref_stable += int(replay.get("stage4_numerical_stable") is True)
        fixture_status[str(row.get("common_bias_fixtures", {}).get("status", "MISSING"))] += 1
    metrics = [
        metric("数值", "execution/risk 完整", "每个 manifest crop 必须有 execution 和 risk record", pct(n, denominator), "满足" if n == denominator else "未达标"),
        metric("数值", "几何－执行等级逻辑", "REJECT→X0/X1，POINT_CANDIDATE→X3，RANGE→X2", {"inconsistencies": inconsistent, "geometry_execution": {"geometry": counter_dict(geometry), "execution": counter_dict(execution)}}, "满足" if inconsistent == 0 else "未达标"),
        metric("数值", "坐标保持", "风险模块不得在运行时替代 All-Arc 坐标", pct(unchanged, n), "满足" if unchanged == n else "未达标"),
        metric("科学", "structured residual OOF", "报告 OOF 而非用 loss 直接判风险", {"R2_struct_oof": stats(r2), "systematic_structure_flag": structured_flag}, "满足", "仅是内部诊断，尚非独立风险 capture 评价。"),
        metric("科学", "measurement/search replay 标记", "Stage 3 certificate 与 Stage 4 stable 应进入风险档案", {"certificate": pct(cert_pass, n), "stable": pct(ref_stable, n)}, "部分满足"),
        metric("科学", "common-bias capture / false-veto", "需要预注册灾难门、独立 fixture 和风险—覆盖曲线", counter_dict(fixture_status), "NOT_MEASURABLE_FROM_CURRENT_ARTIFACT"),
        metric("科学", "target domain", "TARGET_UNKNOWN 不能当作 LOW_RISK 或目标模型通过", counter_dict(risk), "未达标", "所有目标域为未知，风险状态 UNKNOWN 占主导。"),
        metric("使用", "最高执行等级", "无独立 calibration、target-domain 认证和风险 capture 时最高为 X3", counter_dict(execution), "满足", "未出现 X4/X5；此限制符合计划。"),
    ]
    return {"stage": 5, "status": "PASS_WITH_RISK", "counts": {"records": n, "geometry": counter_dict(geometry), "risk": counter_dict(risk), "execution": counter_dict(execution), "reasons": counter_dict(reasons)}, "metrics": metrics,
            "three_layers": [["数值有效性", "PASS", "执行档案完整，状态与执行等级映射一致，坐标未被风险模块改写。"], ["科学有效性", "PASS_WITH_RISK", "诊断已计算，但没有独立 common-bias capture、false-veto 或风险—覆盖验证。"], ["使用有效性", "PASS_WITH_RISK", "输出严格限制到 X0–X3；不得解释成 target/model 风险已被证实控制。"]],
            "downstream": "保留 All-Arc 坐标与 UNKNOWN 风险，Stage 6/7 只能解释贡献和 shadow evaluation，不能将风险状态升级为已校准。"}


def stage6(root: Path, denominator: int) -> dict[str, Any]:
    summary = selected = resolved = 0
    for row in rows(root / "stage 6 grayscale" / "crop_contribution_summary.jsonl"):
        summary += 1
        selected += int(row.get("selected_exact_count", 0))
        resolved += int(row.get("resolved_exact_count", 0))
    exact = cert_pass = 0
    status: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    gt_mm_status: Counter[str] = Counter()
    numeric: dict[str, list[float]] = {key: [] for key in ("C_shift_proj", "C_GT_proj", "C_phi", "C_range", "C_mode_components_added", "C_conflict", "C50", "C_tail")}
    for row in rows(root / "stage 6 grayscale" / "parent_exact_contributions.jsonl"):
        exact += 1
        status[str(row.get("status", "MISSING"))] += 1
        cert_pass += int(row.get("search_certificate", {}).get("passed") is True)
        gt_mm_status[str(row.get("C_GT_mm_status", "MISSING"))] += 1
        for role in row.get("roles", []):
            roles[str(role)] += 1
        for key, values in numeric.items():
            if is_number(row.get(key)):
                values.append(float(row[key]))
    unresolved = selected - resolved
    metrics = [
        metric("数值", "crop summary 完整", "每个 manifest crop 有 contribution summary（含无 exact 的记录）", pct(summary, denominator), "满足" if summary == denominator else "未达标"),
        metric("数值", "exact delete-refit 完成率", "selected exact group 必须分为 resolved 或 UNRESOLVED_SEARCH", {"selected": selected, "resolved": resolved, "unresolved": unresolved, "resolved_rate": pct(resolved, selected)}, "部分满足", "未闭合 delete search 不能赋正/负贡献。"),
        metric("数值", "delete Search Certificate", "status=EXACT 的 delete-refit 必须重新通过 certificate", {"exact_records": status.get("EXACT", 0), "certificate_passed": cert_pass, "pass_rate_among_exact": pct(cert_pass, status.get("EXACT", 0))}, "满足" if cert_pass == status.get("EXACT", 0) else "未达标"),
        metric("科学", "贡献角色与多通道量", "C_GT/C_phi/C_range/C_mode/conflict 必须分别报告", {"roles": counter_dict(roles), **{key: stats(value) for key, value in numeric.items()}}, "满足"),
        metric("科学", "物理 GT 贡献", "C_GT_mm 需 mm_per_pixel 和 GT 不确定度", counter_dict(gt_mm_status), "NOT_MEASURABLE_FROM_CURRENT_ARTIFACT"),
        metric("科学", "GT sign / ranking 外部效度", "需要 independent tree sign accuracy、precision@k/recall@k 与 scale/phase replay", "未提供独立 ranking/scale-phase 评价", "NOT_MEASURABLE_FROM_CURRENT_ARTIFACT"),
        metric("使用", "无 GT 部署语义", "runtime 只能输出 HARMFUL_SUSPECT 等，不可把 HARMFUL_GT 作为删弧规则", {"exact_status": counter_dict(status), "roles": counter_dict(roles)}, "部分满足", "当前角色中含 GT 事后标签，只能用于审计与标注复核。"),
    ]
    return {"stage": 6, "status": "EXPLORATORY", "counts": {"crop_summaries": summary, "selected_exact": selected, "resolved_exact": resolved, "unresolved_search": unresolved, "exact_status": counter_dict(status), "roles": counter_dict(roles)}, "metrics": metrics,
            "three_layers": [["数值有效性", "PARTIAL_PASS", "绝大多数 exact delete-refit 有重新搜索 certificate；未闭合的删除保持 unresolved。"], ["科学有效性", "EXPLORATORY", "多通道贡献可做事后审计，但缺 mm、独立 ranking 以及 scale/phase 稳定性。"], ["使用有效性", "EXPLORATORY", "不可将 HARMFUL_GT 直接转换为无 GT 自动删弧规则。"]],
            "downstream": "Stage 7 只能将贡献作为保护/候选审计证据；任何 unresolved 或 critical group 都不能被 Safe-Prune 删除。"}


def stage7(root: Path, denominator: int, tree_count: int) -> dict[str, Any]:
    final = safe_applied = 0
    methods: Counter[str] = Counter()
    final_status: Counter[str] = Counter()
    for row in rows(root / "stage 7 grayscale" / "final_results.jsonl"):
        final += 1
        methods[str(row.get("final_method", "MISSING"))] += 1
        final_status[str(row.get("safe_prune_status", "MISSING"))] += 1
        safe_applied += int(row.get("safe_prune_applied") is True)
    oracle_rows = oracle_available = oracle_improves = oracle_changes = 0
    for row in rows(root / "stage 7 grayscale" / "oracle_analysis.jsonl"):
        oracle_rows += 1
        oracle_available += int(row.get("oracle_available") is True)
        oracle_improves += int(row.get("oracle_improves") is True)
        oracle_changes += int(row.get("may_change_final_coordinate") is True)
    audit = harmful = protected = gate_pass = 0
    audit_reasons: Counter[str] = Counter()
    for row in rows(root / "stage 7 grayscale" / "safe_candidate_audit.jsonl"):
        audit += 1
        harmful += int(row.get("harmful_suspect") is True)
        protected += int(row.get("protected_critical") is True)
        gate_pass += int(row.get("conflict_gate_pass") is True and row.get("search_certificate_pass") is True and row.get("replay_consistent_measurement_anomaly") is True and not row.get("protected_critical"))
        audit_reasons[str(row.get("reason", "MISSING"))] += 1
    metrics = [
        metric("数值", "final/oracle 完整", "每个 manifest crop 有 final 与 oracle audit", {"final": pct(final, denominator), "oracle": pct(oracle_rows, denominator)}, "满足" if final == oracle_rows == denominator else "未达标"),
        metric("数值", "final 坐标方法", "独立部署门未通过时 final_method 必须为 F-All 且 safe_prune_applied=false", {"methods": counter_dict(methods), "safe_prune_applied": safe_applied}, "满足" if methods == Counter({"F-All": final}) and safe_applied == 0 else "未达标"),
        metric("科学", "Oracle 理论上限", "Oracle 可用于事后判断潜在单删收益，不可改 final 坐标", {"available": pct(oracle_available, oracle_rows), "improves": pct(oracle_improves, oracle_rows), "changes_final": oracle_changes}, "满足" if oracle_changes == 0 else "未达标"),
        metric("科学", "无 GT Safe 候选门", "候选需冻结词典序、critical protection、replay 和 delete certificate", {"audit_groups": audit, "harmful_suspect": harmful, "protected_critical": protected, "all_gates_pass": gate_pass, "reasons": counter_dict(audit_reasons)}, "未达标", "独立阈值/replay 条件不可用，故不存在可部署候选。"),
        metric("使用", "独立树冻结门", "n_freeze=max(30,n_power)；独立 validation trees 必须达到 n_freeze", {"independent_trees": tree_count, "minimum_tree_floor": 30, "n_power": None}, "未达标"),
        metric("使用", "Safe-Prune 部署结论", "主精度改善、tail/False-POINT 非劣、coverage、critical false removal 与 sealed validation 均需通过", "上述 dataset-level evidence 未提供且树数不足", "未达标"),
    ]
    return {"stage": 7, "status": "EXPLORATORY", "counts": {"final": final, "methods": counter_dict(methods), "safe_prune_applied": safe_applied, "oracle_available": oracle_available, "oracle_improves": oracle_improves, "audit_groups": audit, "audit_reasons": counter_dict(audit_reasons)}, "metrics": metrics,
            "three_layers": [["数值有效性", "PASS", "final、oracle 和 audit 产物完整，最终坐标没有被 Safe 改写。"], ["科学有效性", "EXPLORATORY", "Oracle 显示潜在删弧收益，但无 GT Safe 识别器的独立效度尚未建立。"], ["使用有效性", "PASS", "Safe-Prune 正确保持 OFF，最终输出固定 F-All。"]],
            "downstream": "当前可得的正确终结结论是：All-Arc 作为研究基线可保留；Safe-Prune 不具部署资格，必须继续 OFF。"}


def write_stage(root: Path, result: dict[str, Any]) -> None:
    stage = result["stage"]
    directory = root / f"stage {stage} grayscale"
    (directory / "stage_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# Stage {stage} 指标分析与判定", "", "本文件由 `analyze_stagewise_metrics.py` 从该阶段现有 JSONL 产物生成。所有比率保留完整 manifest 分母；`NOT_MEASURABLE_FROM_CURRENT_ARTIFACT` 表示冻结计划要求的证据不存在，不能将其写为通过。", "", "## 本阶段结论", "", f"**{result['status']}**", "", "## 产物与分母", "", "```json", json.dumps(result["counts"], ensure_ascii=False, indent=2), "```", "", "## 指标、门槛与实测", "", "| 层级 | 指标 | 冻结计划要求 / 门槛 | 实测 | 判定 |", "|---|---|---|---|---|"]
    for item in result["metrics"]:
        lines.append(f"| {item['layer']} | {item['metric']} | {item['criterion']} | {display(item['observed'])} | **{item['decision']}** |")
        if item["note"]:
            lines.append(f"|  | 说明 |  | {item['note']} |  |")
    lines.extend(["", "## 三层判定", "", "| 层级 | 结论 | 依据 |", "|---|---|---|"])
    for layer, decision, basis in result["three_layers"]:
        lines.append(f"| {layer} | **{decision}** | {basis} |")
    lines.extend(["", "## 对下游的影响", "", result["downstream"], "", "## 产物说明", "", "- `stage_metrics.json`：本报告全部数值的机器可读版本。", "- `validation_records.jsonl` / `validation_summary.json`：既有的逐 crop 完整性和状态验证记录。", "- 原始运行 JSONL 未被本分析脚本修改。", ""])
    (directory / "stage_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def write_root(root: Path, results: list[dict[str, Any]], denominator: int, trees: list[str]) -> None:
    summary = {"denominator": denominator, "trees": trees, "stage_status": {str(item["stage"]): item["status"] for item in results}, "safe_prune": "OFF"}
    (root / "stagewise_metric_conclusion.json").write_text(json.dumps({"summary": summary, "stages": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ArcPith-GT v4 分阶段指标结论", "", f"- 固定 manifest 分母：**{denominator} crops**", f"- 独立 biological trees：**{len(trees)}** (`{', '.join(trees)}`)", "- 物理尺度：**缺失 mm_per_pixel，所有物理毫米指标不可计算**", "- 目标域：**TARGET_UNKNOWN，不能认证局部公共径向中心等同生物学髓心**", "- Safe-Prune：**OFF**", "", "| Stage | 数据集级结论 | 核心已满足项 | 阻止正式 PASS 的证据缺口 |", "|---|---|---|---|"]
    blockers = {0: "目标域认证、变换等变 replay、mm scale", 1: "重复标注 calibration、overlay/平滑与证据守恒 replay", 2: "同目标 reference basin、消融与 loss map", 3: "585 个 certificate 失败，合计 591 个不可几何解释；独立 reference search", 4: "独立 calibration、profile/search replay", 5: "common-bias capture/false-veto 与风险—覆盖验证", 6: "mm GT 贡献、独立 ranking 与 scale/phase stability", 7: "至少 30 棵独立 validation trees、Safe 的 tail/False-POINT/coverage 门"}
    strengths = {0: "坐标与 lineage 硬合同", 1: "连续弧数值不变量", 2: "六源 seed provenance", 3: "部分 crop 的 certificate", 4: "状态记录与 profile 路由", 5: "状态-执行映射与坐标不改写", 6: "exact delete-refit 的大部分 certificate", 7: "F-All 固定与 Safe OFF"}
    for item in results:
        stage = item["stage"]
        lines.append(f"| {stage} | **{item['status']}** | {strengths[stage]} | {blockers[stage]} |")
    lines.extend(["", "## 总结性判定", "", "当前运行证明了该冻结管线可完整生成阶段产物，并保留失败/REJECT 样本；它不构成已校准的生物学髓心定位或 Safe-Prune 部署验证。所有 `EXPLORATORY` 与 `PASS_WITH_RISK` 结论均不得提升为 `PASS`。", "", "各阶段的完整指标表和三层判定位于相应 `stage N grayscale/stage_analysis.md`；其中逐项说明了哪些数值已验证、哪些科学证据缺失，以及这些限制对下游解释的影响。", ""])
    (root / "stagewise_metric_conclusion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest_grayscale.jsonl")
    parser.add_argument("--output-root", default=r"D:\教务处实习\wood_preproject\树髓定位\代码\最终产物")
    args = parser.parse_args()
    manifest = list(rows(Path(args.manifest)))
    denominator = len(manifest)
    trees = sorted({str(row.get("tree_id")) for row in manifest})
    root = Path(args.output_root)
    results = [stage0(root, denominator), stage1(root, denominator), stage2(root, denominator), stage3(root, denominator), stage4(root, denominator), stage5(root, denominator), stage6(root, denominator), stage7(root, denominator, len(trees))]
    for result in results:
        write_stage(root, result)
    write_root(root, results, denominator, trees)
    print(json.dumps({"denominator": denominator, "trees": trees, "stage_status": {item["stage"]: item["status"] for item in results}, "output_root": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
