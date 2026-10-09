#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIRECTORY}/_common.sh"

CONFIG="${4:-${RACPITH_CONFIG}}"
require_file "${CONFIG}"
MANIFEST="${1:?usage: 00_prepare_grayscale_masks.sh MANIFEST PREPARED_ROOT CLASS_ROOT [CONFIG] [SPLIT_CONFIG]}"
PREPARED_ROOT="${2:?usage: 00_prepare_grayscale_masks.sh MANIFEST PREPARED_ROOT CLASS_ROOT [CONFIG] [SPLIT_CONFIG]}"
CLASS_ROOT="${3:?usage: 00_prepare_grayscale_masks.sh MANIFEST PREPARED_ROOT CLASS_ROOT [CONFIG] [SPLIT_CONFIG]}"
SPLIT_CONFIG="${5:-${RACPITH_PROJECT_ROOT}/configs/urudendro4_split.json}"
require_file "${MANIFEST}"
require_file "${SPLIT_CONFIG}"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/00_prepare_grayscale_masks.py" \
  --manifest "${MANIFEST}" \
  --output "${PREPARED_ROOT}" \
  --class-root "${CLASS_ROOT}" \
  --split-config "${SPLIT_CONFIG}"
