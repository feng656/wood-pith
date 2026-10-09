#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "usage: 40_freeze_after_development.sh INPUT_CONFIG DECISION_RECORD DEVELOPMENT_RUN_ID CODE_REVISION [OUTPUT_CONFIG]" >&2
  echo "prepared/output roots are read from INPUT_CONFIG::paths" >&2
  exit 2
fi

INPUT_CONFIG="$1"
DECISION_RECORD="$2"
DEVELOPMENT_RUN_ID="$3"
CODE_REVISION="$4"
require_file "${INPUT_CONFIG}"
require_file "${DECISION_RECORD}"

PREPARED_ROOT="$(racpith_runtime_path "${INPUT_CONFIG}" prepared_root)"
OUTPUT_ROOT="$(racpith_runtime_path "${INPUT_CONFIG}" output_root)"
OUTPUT_CONFIG="${5:-${OUTPUT_ROOT}/racpith_v1_frozen.json}"
require_directory "${PREPARED_ROOT}"

echo "freezing config=${INPUT_CONFIG}; prepared=${PREPARED_ROOT}; output=${OUTPUT_CONFIG}"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/13_freeze_config.py" \
  --input-config "${INPUT_CONFIG}" \
  --section-manifest "${PREPARED_ROOT}/split/section_manifest.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --reference-manifest "${PREPARED_ROOT}/crops/full_section_reference_manifest.jsonl" \
  --crop-audit "${PREPARED_ROOT}/crops/crop_generation_audit.json" \
  --decision-record "${DECISION_RECORD}" \
  --development-run-id "${DEVELOPMENT_RUN_ID}" \
  --code-revision "${CODE_REVISION}" \
  --output "${OUTPUT_CONFIG}" \
  --acknowledge-no-sealed-results-used

# Validate the newly written lock immediately so a successful freeze command
# always leaves a configuration that the sealed runner can consume.
racpith_py "${RACPITH_PROJECT_ROOT}/scripts/14_validate_frozen_config.py" \
  --config "${OUTPUT_CONFIG}" \
  --section-manifest "${PREPARED_ROOT}/split/section_manifest.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --reference-manifest "${PREPARED_ROOT}/crops/full_section_reference_manifest.jsonl" \
  --crop-audit "${PREPARED_ROOT}/crops/crop_generation_audit.json" \
  --decision-record "${DECISION_RECORD}"
