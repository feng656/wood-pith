#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG="${3:-${RACPITH_CONFIG}}"
require_file "${CONFIG}"
PREPARED_ROOT="${1:-$(racpith_runtime_path "${CONFIG}" prepared_root)}"
RUN_ROOT="${2:-$(racpith_runtime_path "${CONFIG}" development_root)}"
WORKERS="${4:-${RACPITH_WORKERS:-1}}"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/05_run_contributions.py" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --workers "${WORKERS}" \
  --output "${RUN_ROOT}/contributions"
