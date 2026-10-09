"""
汇总裁剪矩形的髓心新坐标
========================
理论：裁剪后图像以左上角为原点，新髓心坐标 = 原髓心坐标 - 裁剪起点坐标
即：cx_new = cx_global - bx,  cy_new = cy_global - by

从已有的 patch JSON 标注中提取并汇总到单独目录。
"""
import json
import pandas as pd
from pathlib import Path
import numpy as np

PATCH_ROOT = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")
OUT_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\矩形新的髓心坐标")
OUT_DIR.mkdir(exist_ok=True, parents=True)

rows = []
bad_files = []
json_files = sorted(PATCH_ROOT.glob("*/*.json"))
print(f"找到 {len(json_files)} 个 JSON 标注文件")

for jf in json_files:
    try:
        with open(jf, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except Exception as e:
        bad_files.append((str(jf), str(e)))
        continue

    rows.append({
        'patch_name':     d['patch_name'],
        'sample':         d['sample'],
        'img_size':       d['size'],
        # 裁剪区域在原图的位置
        'bx':             d['bx'],
        'by':             d['by'],
        # 原图上的髓心坐标（原点=原图左上角）
        'cx_original':    d['cx_global'],
        'cy_original':    d['cy_global'],
        # 裁剪后的新髓心坐标（原点=裁剪图左上角）
        'cx_new':         d['cx_local'],
        'cy_new':         d['cy_local'],
        # 验证公式：cx_new = cx_original - bx
        'formula_check_x': d['cx_global'] - d['bx'],
        'formula_check_y': d['cy_global'] - d['by'],
        # 髓心是否在裁剪图内
        'pith_in_patch':  d['pith_in_patch'],
        # 其他指标
        'dist_to_center': d['dist_to_pith'],
        'angle_span_deg': d['angle_span_deg'],
        'n_ring_arcs':    d['n_ring_arcs'],
        'n_ring_points':  d['n_ring_points'],
    })

df = pd.DataFrame(rows)

# 验证公式正确性（应该全部为 0）
x_ok = (df['cx_new'] == df['formula_check_x']).all()
y_ok = (df['cy_new'] == df['formula_check_y']).all()
print(f"公式验证 cx_new = cx_original - bx : {'✅ 正确' if x_ok else '❌ 有误'}")
print(f"公式验证 cy_new = cy_original - by : {'✅ 正确' if y_ok else '❌ 有误'}")

# 删除验证列，保持输出整洁
df = df.drop(columns=['formula_check_x', 'formula_check_y'])

# 保存 CSV（主文件，方便 Excel 打开）
csv_path = OUT_DIR / "全部矩形髓心坐标.csv"
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# 也保存一份 JSON Lines 格式（方便程序读取）
jsonl_path = OUT_DIR / "全部矩形髓心坐标.jsonl"
with open(jsonl_path, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        f.write(row.to_json(force_ascii=False) + '\n')

# ==================== 统计 ====================
in_patch  = df[df['pith_in_patch'] == True]
out_patch = df[df['pith_in_patch'] == False]

print(f"\n{'='*60}")
print(f"汇总完毕！")
print(f"  总块数:          {len(df)}")
print(f"  髓心在块内:      {len(in_patch)} 块")
print(f"  髓心在块外:      {len(out_patch)} 块")
print(f"  平均距离(中心):  {df['dist_to_center'].mean():.0f} px")
print(f"  平均角度跨度:    {df['angle_span_deg'].mean():.0f} deg")
print(f"  平均年轮弧段:    {df['n_ring_arcs'].mean():.1f}")
print(f"\n  输出目录:        {OUT_DIR}")
print(f"  CSV 文件:        {csv_path}")
print(f"  JSONL 文件:      {jsonl_path}")

if bad_files:
    print(f"\n!! 损坏/无法解析的 JSON ({len(bad_files)} 个):")
    for bf, err in bad_files[:10]:
        print(f"  {bf}")
        print(f"    错误: {err}")
