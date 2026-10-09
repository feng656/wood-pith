#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "usage: 50_run_sealed_once.sh FROZEN_CONFIG RUN_ID DECISION_RECORD [WORKERS]" >&2
  echo "prepared/sealed roots are read from FROZEN_CONFIG::paths" >&2
  exit 2
fi

CONFIG="$1"
RUN_ID="$2"
DECISION_RECORD="$3"
WORKERS="${4:-${RACPITH_WORKERS:-1}}"
require_file "${CONFIG}"
require_file "${DECISION_RECORD}"

PREPARED_ROOT="$(racpith_runtime_path "${CONFIG}" prepared_root)"
RUN_ROOT="$(racpith_runtime_path "${CONFIG}" sealed_root)"
require_directory "${PREPARED_ROOT}"

echo "sealed config=${CONFIG}; prepared=${PREPARED_ROOT}; output=${RUN_ROOT}; run_id=${RUN_ID}"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/14_validate_frozen_config.py" \
  --config "${CONFIG}" \
  --section-manifest "${PREPARED_ROOT}/split/section_manifest.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --reference-manifest "${PREPARED_ROOT}/crops/full_section_reference_manifest.jsonl" \
  --crop-audit "${PREPARED_ROOT}/crops/crop_generation_audit.json" \
  --decision-record "${DECISION_RECORD}"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/15_open_sealed_test.py" \
  --config "${CONFIG}" \
  --run-id "${RUN_ID}" \
  --marker "${RUN_ROOT}/sealed_test_opening.json"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/03_build_evidence.py" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --workers "${WORKERS}" \
  --resume \
  --output "${RUN_ROOT}/evidence"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/03_build_evidence.py" \
  --crop-manifest "${PREPARED_ROOT}/crops/full_section_reference_manifest.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --workers "${WORKERS}" \
  --resume \
  --output "${RUN_ROOT}/target_reference/evidence"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/04_run_localization.py" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --workers "${WORKERS}" \
  --resume \
  --run-id "${RUN_ID}" \
  --output "${RUN_ROOT}/core"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/04_run_localization.py" \
  --evidence-index "${RUN_ROOT}/target_reference/evidence/evidence_index.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --workers "${WORKERS}" \
  --resume \
  --run-id "${RUN_ID}:target-reference" \
  --output "${RUN_ROOT}/target_reference/core"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/05b_run_uncertainty.py" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --workers "${WORKERS}" \
  --resume \
  --output "${RUN_ROOT}/uncertainty"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/05_run_contributions.py" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --workers "${WORKERS}" \
  --output "${RUN_ROOT}/contributions"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/05_run_contributions.py" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --include-audit-partitions \
  --replays "${RACPITH_PARTITION_AUDIT_REPLAYS:-16}" \
  --skip-crossfit \
  --max-crops-per-tree "${RACPITH_PARTITION_AUDIT_CROPS_PER_TREE:-1}" \
  --workers "${WORKERS}" \
  --output "${RUN_ROOT}/contribution_partition_audit"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/11_run_baselines.py" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --output "${RUN_ROOT}/baselines/baselines.jsonl"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/06d_assess_target_domain.py" \
  --reference-manifest "${PREPARED_ROOT}/crops/full_section_reference_manifest.jsonl" \
  --predictions "${RUN_ROOT}/target_reference/core/predictions" \
  --result-index "${RUN_ROOT}/target_reference/core/result_index.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --output "${RUN_ROOT}/analysis/target_domain.jsonl"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/07_evaluate.py" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --predictions "${RUN_ROOT}/core/predictions" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --uncertainty-index "${RUN_ROOT}/uncertainty/uncertainty_index.jsonl" \
  --target-domain-index "${RUN_ROOT}/analysis/target_domain.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --output "${RUN_ROOT}/analysis"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/12_evaluate_baselines.py" \
  --baselines "${RUN_ROOT}/baselines/baselines.jsonl" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --output "${RUN_ROOT}/baseline_analysis"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/07c_evaluate_runtime.py" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --evidence-index "${RUN_ROOT}/evidence/evidence_index.jsonl" \
  --result-index "${RUN_ROOT}/core/result_index.jsonl" \
  --uncertainty-index "${RUN_ROOT}/uncertainty/uncertainty_index.jsonl" \
  --contribution-index "${RUN_ROOT}/contributions/contribution_index.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --output "${RUN_ROOT}/analysis"

if [[ -n "${RACPITH_CALIBRATOR_MODEL:-}" && -n "${RACPITH_CALIBRATOR_METADATA:-}" ]]; then
  racpith_py "${RACPITH_PROJECT_ROOT}/scripts/06b_apply_contribution_calibrator.py" \
    --contributions "${RUN_ROOT}/contributions" \
    --contribution-index "${RUN_ROOT}/contributions/contribution_index.jsonl" \
    --model "${RACPITH_CALIBRATOR_MODEL}" \
    --metadata "${RACPITH_CALIBRATOR_METADATA}" \
    --config "${CONFIG}" \
    --split sealed_test \
    --output "${RUN_ROOT}/contribution_calibrator_evaluation"
fi

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
  --split sealed_test \
  --output "${RUN_ROOT}/contribution_analysis"

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/08b_visualize_contributions.py" \
  --contributions "${RUN_ROOT}/contributions" \
  --contribution-index "${RUN_ROOT}/contributions/contribution_index.jsonl" \
  --crop-manifest "${PREPARED_ROOT}/crops/crop_manifest.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
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
  --split sealed_test \
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
  --split sealed_test \
  --output "${RUN_ROOT}/audit/partition_contribution_audit.json" || AUDIT_STATUS=$?

racpith_py "${RACPITH_PROJECT_ROOT}/scripts/09_audit.py" \
  --source-root "${RACPITH_PROJECT_ROOT}/src/racpith" \
  --crop-manifest "${PREPARED_ROOT}/crops/full_section_reference_manifest.jsonl" \
  --evidence-index "${RUN_ROOT}/target_reference/evidence/evidence_index.jsonl" \
  --predictions "${RUN_ROOT}/target_reference/core/predictions" \
  --result-index "${RUN_ROOT}/target_reference/core/result_index.jsonl" \
  --target-domain-index "${RUN_ROOT}/analysis/target_domain.jsonl" \
  --config "${CONFIG}" \
  --split sealed_test \
  --output "${RUN_ROOT}/audit/target_reference_audit.json" || AUDIT_STATUS=$?

if [[ "${AUDIT_STATUS}" -ne 0 ]]; then
  echo "sealed audit failed; no sealed report was generated and no PASS claim is allowed" >&2
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
  --output "${RUN_ROOT}/RAC_Pith_sealed_test_report.md"
