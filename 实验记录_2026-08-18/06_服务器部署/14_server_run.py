"""服务器一键运行：训练（GPU，自动 CUDA）→ 24 张 val 评估 → 汇总。

与本地小数据实验同参数（120 样本 · 384 画布 · 10 轮），
geometry 使用官方档预算；后验空张量 bug 已修复（posterior.py）。
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

from oapith.workflows import run_training

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_CONFIG = PROJECT_ROOT / "configs/urudendro_server.yaml"
RESUME_CHECKPOINT: Path | None = None   # 断点续跑时填 runs/oa_pith_server/best.pt


def main() -> None:
    os.chdir(PROJECT_ROOT)
    print("== 1) 训练（GPU，无则回退 CPU）==")
    run_training(SERVER_CONFIG, resume=RESUME_CHECKPOINT, device=None)
    print("== 2) 评估 24 张 val ==")
    os.environ["OA_PITH_EVAL_CONFIG"] = str(SERVER_CONFIG)
    runpy.run_path(str(PROJECT_ROOT / "scripts/13_eval_small.py"), run_name="__main__")


if __name__ == "__main__":
    main()
