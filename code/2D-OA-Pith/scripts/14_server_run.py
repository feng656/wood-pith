"""服务器一键运行：训练（GPU，自动 CUDA）→ 24 张 val 评估 → 汇总。

与本地小数据实验同参数（120 样本 · 384 画布 · 10 轮），
geometry 使用官方档预算；后验空张量 bug 已修复（posterior.py）。
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:   # 免 pip install：直接引入 src
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oapith.workflows import run_training  # noqa: E402

SERVER_CONFIG = PROJECT_ROOT / "configs/urudendro_server.yaml"
RESUME_CHECKPOINT: Path | None = None   # 断点续跑时填 runs/oa_pith_server/best.pt
# 显卡选择：OA_PITH_DEVICE=cuda:1（或 CUDA_VISIBLE_DEVICES=1）；默认 None = 自动 cuda:0
DEVICE = os.environ.get("OA_PITH_DEVICE") or None


def main() -> None:
    os.chdir(PROJECT_ROOT)
    print(f"== 1) 训练（device={DEVICE or 'auto(cuda 优先)'}）==")
    run_training(SERVER_CONFIG, resume=RESUME_CHECKPOINT, device=DEVICE)
    print("== 2) 评估 24 张 val ==")
    os.environ["OA_PITH_EVAL_CONFIG"] = str(SERVER_CONFIG)
    runpy.run_path(str(PROJECT_ROOT / "scripts/13_eval_small.py"), run_name="__main__")


if __name__ == "__main__":
    main()
