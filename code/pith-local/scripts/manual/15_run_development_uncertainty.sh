#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG="${2:-${RACPITH_CONFIG}}"
require_file "${CONFIG}"
RUN_ROOT="${1:-$(racpith_runtime_path "${CONFIG}" development_root)}"
WORKERS="${3:-${RACPITH_WORKERS:-1}}"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/05b_run_uncertainty.py" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --workers "${WORKERS}" \
  --output "${RUN_ROOT}/uncertainty"
