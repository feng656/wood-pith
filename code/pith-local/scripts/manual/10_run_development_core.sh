#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG="${3:-${RACPITH_CONFIG}}"
require_file "${CONFIG}"
PREPARED_ROOT="${1:-$(racpith_runtime_path "${CONFIG}" prepared_root)}"
RUN_ROOT="${2:-$(racpith_runtime_path "${CONFIG}" development_root)}"
if [[ -n "${4:-}" ]]; then
  WORKERS="$4"
elif [[ -n "${RACPITH_WORKERS:-}" ]]; then
  WORKERS="${RACPITH_WORKERS}"
else
  WORKERS="$(nproc 2>/dev/null || echo 4)"
  if [[ "${WORKERS}" -gt 16 ]]; then WORKERS=16; fi
fi
if [[ "${WORKERS}" -gt 1 ]]; then
  # One process per crop already provides concurrency; avoid BLAS thread
  # oversubscription across workers.
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
fi
RUN_ID="${RACPITH_DEVELOPMENT_RUN_ID:-racpith-development}"

require_file "${PREPARED_ROOT}/crops/crop_manifest.jsonl"
require_file "${CONFIG}"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/03_build_evidence.py" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --workers "${WORKERS}" \
  --resume \
  --output "${RUN_ROOT}/evidence"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/04_run_localization.py" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --workers "${WORKERS}" \
  --run-id "${RUN_ID}" \
  --resume \
  --output "${RUN_ROOT}/core"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/11_run_baselines.py" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --output "${RUN_ROOT}/baselines/baselines.jsonl"

# Full-section reference is fitted without GT.  Anatomical-pith comparison is
# deferred to the evaluation stage.
racpith_py "${RACPITH_PROJECT_ROOT}/scripts/03_build_evidence.py" \
  --crop-manifest "${PREPARED_ROOT}/crops/full_section_reference_manifest.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --workers "${WORKERS}" \
  --resume \
  --output "${RUN_ROOT}/target_reference/evidence"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/04_run_localization.py" \
  --evidence-index "${RUN_ROOT}/target_reference/evidence/evidence_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --workers "${WORKERS}" \
  --run-id "${RUN_ID}:target-reference" \
  --resume \
  --output "${RUN_ROOT}/target_reference/core"
