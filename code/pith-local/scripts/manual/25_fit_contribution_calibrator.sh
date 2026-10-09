#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG="${2:-${RACPITH_CONFIG}}"
require_file "${CONFIG}"
RUN_ROOT="${1:-$(racpith_runtime_path "${CONFIG}" development_root)}"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/06_calibrate_contributions.py" \
  --contributions "${RUN_ROOT}/contributions" \
  --contribution-index "${RUN_ROOT}/contributions/contribution_index.jsonl" \
  --config "${CONFIG}" \
  --train-split train \
  --folds 6 \
  --output "${RUN_ROOT}/contribution_calibrator"
