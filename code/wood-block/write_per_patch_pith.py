"""
给每个 patch 写一个独立的髓心坐标文件
======================================
格式：{patch_name}_新髓心.json
放在和 JPG 同级目录下，一目了然
"""
import json
from pathlib import Path

PATCH_ROOT = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")

json_files = sorted(PATCH_ROOT.glob("*/*.json"))
total = 0

for jf in json_files:
    # 跳过已生成的髓心文件
    if jf.stem.endswith("_新髓心"):
        continue

    with open(jf, 'r', encoding='utf-8') as f:
        d = json.load(f)

    new_file = jf.parent / f"{d['patch_name']}_新髓心.json"

    coord = {
        'patch': d['patch_name'],
        'sample': d['sample'],
        'img_size': d['size'],
        '原图髓心': [d['cx_global'], d['cy_global']],
        '裁剪起点': [d['bx'], d['by']],
        '新髓心坐标': [d['cx_local'], d['cy_local']],
        '髓心在块内': d['pith_in_patch'],
        '距离块中心': d['dist_to_pith'],
    }

    with open(new_file, 'w', encoding='utf-8') as f:
        json.dump(coord, f, ensure_ascii=False, indent=2)
    total += 1

print(f"完成！生成了 {total} 个 _新髓心.json 文件")
print(f"示例目录: {json_files[0].parent}")
