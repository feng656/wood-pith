#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG="${3:-${RACPITH_CONFIG}}"
require_file "${CONFIG}"
PREPARED_ROOT="${1:-$(racpith_runtime_path "${CONFIG}" prepared_root)}"
RUN_ROOT="${2:-$(racpith_runtime_path "${CONFIG}" development_root)}"
WORKERS="${4:-${RACPITH_WORKERS:-1}}"
REPLAYS="${RACPITH_PARTITION_AUDIT_REPLAYS:-16}"
CROPS_PER_TREE="${RACPITH_PARTITION_AUDIT_CROPS_PER_TREE:-1}"

# This is a separate sensitivity audit.  It never changes the All-Arc estimate,
# primary contribution registry, or frozen operating point.
racpith_py "${RACPITH_PROJECT_ROOT}/scripts/05_run_contributions.py" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --include-audit-partitions \
  --replays "${REPLAYS}" \
  --skip-crossfit \
  --max-crops-per-tree "${CROPS_PER_TREE}" \
  --workers "${WORKERS}" \
  --output "${RUN_ROOT}/contribution_partition_audit"
