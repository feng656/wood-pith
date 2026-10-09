#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIRECTORY}/_common.sh"

if [[ "$#" -ne 0 ]]; then
  echo "run_all_development.sh accepts no positional paths; edit configs/racpith_v1.json::paths or set RACPITH_CONFIG" >&2
  exit 2
fi

CONFIG="${RACPITH_CONFIG}"
require_file "${CONFIG}"
CONFIG_DATASET_ROOT="$(racpith_runtime_path "${CONFIG}" dataset_root)"
CONFIG_PREPARED_ROOT="$(racpith_runtime_path "${CONFIG}" prepared_root)"
CONFIG_DEVELOPMENT_ROOT="$(racpith_runtime_path "${CONFIG}" development_root)"
DATASET_ROOT="${CONFIG_DATASET_ROOT}"
PREPARED_ROOT="${CONFIG_PREPARED_ROOT}"
RUN_ROOT="${CONFIG_DEVELOPMENT_ROOT}"
if [[ -n "${RACPITH_WORKERS:-}" ]]; then
  WORKERS="${RACPITH_WORKERS}"
else
  WORKERS="$(nproc 2>/dev/null || echo 4)"
  if [[ "${WORKERS}" -gt 16 ]]; then WORKERS=16; fi
fi
PITH_ORDER="${RACPITH_PITH_ORDER:-auto}"

echo "runtime config=${CONFIG}; dataset=${DATASET_ROOT}; prepared=${PREPARED_ROOT}; development=${RUN_ROOT}"

bash "${SCRIPT_DIRECTORY}/00_prepare_data.sh" \
  "${DATASET_ROOT}" \
  "${PREPARED_ROOT}" \
  "${PITH_ORDER}" \
  "${CONFIG}"
bash "${SCRIPT_DIRECTORY}/10_run_development_core.sh" "${PREPARED_ROOT}" "${RUN_ROOT}" "${CONFIG}" "${WORKERS}"
bash "${SCRIPT_DIRECTORY}/15_run_development_uncertainty.sh" "${RUN_ROOT}" "${CONFIG}" "${WORKERS}"
bash "${SCRIPT_DIRECTORY}/20_run_development_contributions.sh" "${PREPARED_ROOT}" "${RUN_ROOT}" "${CONFIG}" "${WORKERS}"
bash "${SCRIPT_DIRECTORY}/22_run_development_partition_audit.sh" "${PREPARED_ROOT}" "${RUN_ROOT}" "${CONFIG}" "${WORKERS}"
bash "${SCRIPT_DIRECTORY}/30_analyze_development.sh" "${PREPARED_ROOT}" "${RUN_ROOT}" "${CONFIG}"

echo "development workflow complete; sealed_test has not been opened"
