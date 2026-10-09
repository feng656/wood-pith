"""Build the JSONL manifest from the UruDendro cropped-rectangle patch dataset.

每张 patch 的 {patch}.json 直接给出本工程所需的全部几何：
- rings[].points —— patch 局部像素坐标的年轮 polyline（labelme 标注）
- cx_local/cy_local —— patch 局部像素坐标的髓心（= 原图髓心 − 裁剪起点，允许在图外）
- dist_to_pith / angle_span_deg —— 观测可辨识性参考（写入 metadata，不参与监督）

坐标约定与原工程一致：左上像素中心 (0,0)，pith 坐标不裁剪。
本数据集无物理标定（无 mm/px），故不写 mm_per_pixel；误差按像素/相对图宽报告。

group_id 取样本名的 T# 前缀（树级最高依赖层），保证同树（含 _ADAP 变体）
全部在同一 split；split 按 T# 组级划分，固定确定性顺序。
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ── Edit these ────────────────────────────────────────────────
SOURCE_DIR = Path("D:/教务处实习/wood_preproject/树髓定位/裁剪矩形")
OUTPUT_MANIFEST = Path(__file__).resolve().parents[1] / "data/manifest.jsonl"
SMOKE_MANIFEST = Path(__file__).resolve().parents[1] / "data/manifest_smoke.jsonl"
IMAGE_PREFIX = "裁剪矩形"          # image 路径相对 data.root（见 configs/urudendro_smoke.yaml）
SMOKE_TRAIN_GROUPS = 2             # 冒烟子集：前 2 个 train 组 + 前 1 个 val 组
SMOKE_VAL_GROUPS = 1
SMOKE_MAX_PER_GROUP = 12           # 冒烟子集每组最多取多少个 patch
TRAIN_FRACTION = 0.8               # 组级 train 占比（组数，按 T# 前缀排序后切分）
MIN_CURVE_POINTS = 3               # 与 RingCurve 校验一致（≥3 点、无连续重复）
CURVE_SIGMA_PX = 1.5               # labelme 多边形标注精度假设（像素）
PITH_SIGMA_PX = 8.0                # 髓心真值来自圆拟合/人工标注，σ≈8px 假设
# ───────────────────────────────────────────────────────────────

GROUP_PATTERN = re.compile(r"^(T\d+)_")


def build_records() -> tuple[list[dict], Counter[str], Counter[str]]:
    records: list[dict] = []
    per_group: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for sample_dir in sorted(SOURCE_DIR.iterdir()):
        if not sample_dir.is_dir():
            continue
        for json_path in sorted(sample_dir.glob("*.json")):
            if json_path.name.endswith("_新髓心.json"):
                continue
            try:
                ann = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                skipped[f"json_error:{exc.__class__.__name__}"] += 1
                continue
            patch = str(ann.get("patch_name") or json_path.stem)
            sample = str(ann.get("sample") or sample_dir.name)
            image_path = sample_dir / f"{patch}.jpg"
            if not image_path.is_file():
                skipped["missing_jpg"] += 1
                continue
            rings = ann.get("rings") or []
            curves = []
            for ring in rings:
                points = ring.get("points") or []
                if len(points) < MIN_CURVE_POINTS:
                    continue
                clean: list[list[float]] = []
                for point in points:
                    value = [float(point[0]), float(point[1])]
                    if clean and clean[-1] == value:
                        continue  # RingCurve 校验拒绝连续重复点
                    clean.append(value)
                if len(clean) < MIN_CURVE_POINTS:
                    continue
                curves.append({
                    "ring_id": int(ring["ring_idx"]),
                    "points_px": clean,
                    "sigma_px": CURVE_SIGMA_PX,
                    "visibility": 1.0,
                    "annotator_id": "labelme_v1",
                    "arc_id": f"r{int(ring['ring_idx'])}",
                })
            if not curves:
                skipped["no_valid_curves"] += 1
                continue
            match = GROUP_PATTERN.match(sample)
            group_id = match.group(1) if match else sample
            dist = float(ann.get("dist_to_pith") or 0.0)
            size = float(ann.get("size") or max(float(ann.get("img_W", 0)), float(ann.get("img_H", 0))))
            # near/far 与 PithDataset 训练期重推导一致：‖pith‖≤switch_radius(2.0)
            # 等价于 距离≤边长（方形 patch）。infinity/null 不分配（README：仅可辨识
            # 性实验或严格合成才监督这两个状态）。
            pith_state = 0 if dist <= size else 1
            records.append({
                "sample_id": patch,
                "group_id": group_id,
                "split": None,  # 组级划分，下面统一赋值
                "image": f"{IMAGE_PREFIX}/{sample}/{patch}.jpg",
                "metadata": {
                    "disc_id": sample,
                    "size_px": int(size),
                    "bx": ann.get("bx"), "by": ann.get("by"),
                    "cx_global": ann.get("cx_global"), "cy_global": ann.get("cy_global"),
                    "pith_in_patch": bool(ann.get("pith_in_patch")),
                    "dist_to_pith": dist,
                    "angle_span_deg": ann.get("angle_span_deg"),
                    "n_ring_arcs": len(curves),
                },
                "pith_px": [float(ann["cx_local"]), float(ann["cy_local"])],
                "pith_cov_px": [[PITH_SIGMA_PX ** 2, 0.0], [0.0, PITH_SIGMA_PX ** 2]],
                "pith_state": pith_state,
                "rings_annotated": True,
                "curves": curves,
            })
            per_group[group_id] += 1
    return records, per_group, skipped


def assign_splits(records: list[dict]) -> dict[str, str]:
    groups = sorted({record["group_id"] for record in records})
    split_count = max(1, int(round(len(groups) * TRAIN_FRACTION)))
    mapping = {group: ("train" if index < split_count else "validation") for index, group in enumerate(groups)}
    return mapping


def dump(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    records, per_group, skipped = build_records()
    split_of = assign_splits(records)
    for record in records:
        record["split"] = split_of[record["group_id"]]

    split_counts: Counter[str] = Counter()
    for record in records:
        split_counts[record["split"]] += 1
    print(f"有效样本 {len(records)} 个 · 组 {len(per_group)} 个")
    print("各组样本数:", dict(sorted(per_group.items())))
    print("split 分布:", dict(split_counts))
    print("跳过原因:", dict(skipped))
    states = Counter(record["pith_state"] for record in records)
    print("pith_state 分布 (0=near,1=far):", dict(states))
    in_patch = sum(1 for record in records if record["metadata"]["pith_in_patch"])
    print(f"髓心在图内: {in_patch}/{len(records)}")
    print(f"每样本曲线数: min={min(len(r['curves']) for r in records)} "
          f"max={max(len(r['curves']) for r in records)}")

    # 冒烟子集：前 N 个 train 组 + 前 M 个 val 组（按组名排序，确定性），每组限量
    groups_by_split: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups_by_split[record["split"]].append(record["group_id"])
    smoke_groups = (
        sorted(set(groups_by_split["train"]))[:SMOKE_TRAIN_GROUPS]
        + sorted(set(groups_by_split["validation"]))[:SMOKE_VAL_GROUPS]
    )
    smoke_records = []
    for group in smoke_groups:
        smoke_records.extend(
            [record for record in records if record["group_id"] == group][:SMOKE_MAX_PER_GROUP]
        )
    smoke_groups_split = {g: split_of[g] for g in smoke_groups}
    print(f"冒烟子集: {len(smoke_records)} 样本 · 组 {smoke_groups_split}")

    dump(OUTPUT_MANIFEST, records)
    dump(SMOKE_MANIFEST, smoke_records)

    # 用官方 loader 验证（唯一 sample_id、组不跨 split、字段校验）
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from oapith.data.manifest import load_manifest
    loaded = load_manifest(SMOKE_MANIFEST)
    print(f"load_manifest 校验通过：{len(loaded)} 条（schema + 组级 split 检查）✓")
    full = load_manifest(OUTPUT_MANIFEST)
    print(f"全量 manifest 校验通过：{len(full)} 条 ✓")
    print(f"\n输出: {OUTPUT_MANIFEST}\n      {SMOKE_MANIFEST}")


if __name__ == "__main__":
    main()
