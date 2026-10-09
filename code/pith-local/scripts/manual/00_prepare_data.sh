#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG="${4:-${RACPITH_CONFIG}}"
require_file "${CONFIG}"
CONFIG_DATASET_ROOT="$(racpith_runtime_path "${CONFIG}" dataset_root)"
CONFIG_PREPARED_ROOT="$(racpith_runtime_path "${CONFIG}" prepared_root)"
DATASET_ROOT="${1:-${CONFIG_DATASET_ROOT}}"
RUN_ROOT="${2:-${CONFIG_PREPARED_ROOT}}"
PITH_ORDER="${3:-auto}"
SPLIT_CONFIG="${RACPITH_SPLIT_CONFIG:-${RACPITH_PROJECT_ROOT}/configs/urudendro4_split.json}"
CROP_CONFIG="${RACPITH_CROP_CONFIG:-${RACPITH_PROJECT_ROOT}/configs/urudendro4_crops.json}"

require_directory "${DATASET_ROOT}"
require_file "${SPLIT_CONFIG}"
require_file "${CROP_CONFIG}"

bash "$(dirname "$0")/00_check_environment.sh" "${RUN_ROOT}/environment.json"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/00_index_dataset.py" \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --pith-order "${PITH_ORDER}" \
  --output "${RUN_ROOT}/data_index"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/01_split_by_tree.py" \
  --source-manifest "${RUN_ROOT}/data_index/source_manifest.jsonl" \
  --config "${SPLIT_CONFIG}" \
  --output "${RUN_ROOT}/split"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/02_generate_crops.py" \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --section-manifest "${RUN_ROOT}/split/section_manifest.jsonl" \
  --crop-config "${CROP_CONFIG}" \
  --pith-order "${PITH_ORDER}" \
  --resume \
  --output "${RUN_ROOT}/crops"

echo "prepared manifests and derived crops under ${RUN_ROOT}; source dataset was read-only"
