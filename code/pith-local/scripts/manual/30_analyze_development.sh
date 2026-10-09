#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG="${3:-${RACPITH_CONFIG}}"
require_file "${CONFIG}"
PREPARED_ROOT="${1:-$(racpith_runtime_path "${CONFIG}" prepared_root)}"
RUN_ROOT="${2:-$(racpith_runtime_path "${CONFIG}" development_root)}"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/06d_assess_target_domain.py" \
  --reference-manifest "${PREPARED_ROOT}/crops/full_section_reference_manifest.jsonl" \
  --predictions "${RUN_ROOT}/target_reference/core/predictions" \
  --result-index "${RUN_ROOT}/target_reference/core/result_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --output "${RUN_ROOT}/analysis/target_domain.jsonl"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/07_evaluate.py" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --uncertainty-index "${RUN_ROOT}/uncertainty/uncertainty_index.jsonl" \
  --target-domain-index "${RUN_ROOT}/analysis/target_domain.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --output "${RUN_ROOT}/analysis"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/12_evaluate_baselines.py" \
  --baselines "${RUN_ROOT}/baselines/baselines.jsonl" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --output "${RUN_ROOT}/baseline_analysis"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/07c_evaluate_runtime.py" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --uncertainty-index "${RUN_ROOT}/uncertainty/uncertainty_index.jsonl" \
  --contribution-index "${RUN_ROOT}/contributions/contribution_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --output "${RUN_ROOT}/analysis"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/08_visualize.py" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --metrics "${RUN_ROOT}/analysis/crop_metrics.csv" \
  --uncertainty-index "${RUN_ROOT}/uncertainty/uncertainty_index.jsonl" \
  --target-domain-index "${RUN_ROOT}/analysis/target_domain.jsonl" \
  --config "${CONFIG}" \
  --output "${RUN_ROOT}/figures"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/07b_evaluate_contributions.py" \
  --contributions "${RUN_ROOT}/contributions" \
  --contribution-index "${RUN_ROOT}/contributions/contribution_index.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --partition-audit-contributions "${RUN_ROOT}/contribution_partition_audit" \
  --partition-audit-index "${RUN_ROOT}/contribution_partition_audit/contribution_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --output "${RUN_ROOT}/contribution_analysis"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/08b_visualize_contributions.py" \
  --contributions "${RUN_ROOT}/contributions" \
  --contribution-index "${RUN_ROOT}/contributions/contribution_index.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --output "${RUN_ROOT}/contribution_figures"

AUDIT_STATUS=0
racpith_py "${RACPITH_PROJECT_ROOT}/scripts/09_audit.py" \
  --source-root "${RACPITH_PROJECT_ROOT}/src/racpith" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --uncertainty-index "${RUN_ROOT}/uncertainty/uncertainty_index.jsonl" \
  --contributions "${RUN_ROOT}/contributions" \
  --contribution-index "${RUN_ROOT}/contributions/contribution_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --output "${RUN_ROOT}/audit/combined_audit.json" || AUDIT_STATUS=$?

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/09_audit.py" \
  --source-root "${RACPITH_PROJECT_ROOT}/src/racpith" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --contributions "${RUN_ROOT}/contribution_partition_audit" \
  --contribution-index "${RUN_ROOT}/contribution_partition_audit/contribution_index.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --output "${RUN_ROOT}/audit/partition_contribution_audit.json" || AUDIT_STATUS=$?

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/09_audit.py" \
  --source-root "${RACPITH_PROJECT_ROOT}/src/racpith" \
  --crop-manifest "${PREPARED_ROOT}/crops/full_section_reference_manifest.jsonl" \
  --evidence-index "${RUN_ROOT}/target_reference/evidence/evidence_index.jsonl" \
  --predictions "${RUN_ROOT}/target_reference/core/predictions" \
  --result-index "${RUN_ROOT}/target_reference/core/result_index.jsonl" \
  --target-domain-index "${RUN_ROOT}/analysis/target_domain.jsonl" \
  --config "${CONFIG}" \
  --split train \
  --output "${RUN_ROOT}/audit/target_reference_audit.json" || AUDIT_STATUS=$?

if [[ "${AUDIT_STATUS}" -ne 0 ]]; then
  echo "audit failed; no sealed development report was generated; inspect ${RUN_ROOT}/audit" >&2
  exit "${AUDIT_STATUS}"
fi

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/10_build_report.py" \
  --metrics "${RUN_ROOT}/analysis" \
  --figures "${RUN_ROOT}/figures" \
  --contribution-metrics "${RUN_ROOT}/contribution_analysis" \
  --contribution-figures "${RUN_ROOT}/contribution_figures" \
  --baseline-metrics "${RUN_ROOT}/baseline_analysis" \
  --audit "${RUN_ROOT}/audit/combined_audit.json" \
  --audit "${RUN_ROOT}/audit/partition_contribution_audit.json" \
  --audit "${RUN_ROOT}/audit/target_reference_audit.json" \
  --output "${RUN_ROOT}/RAC_Pith_development_report.md"
