#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

OUTPUT="${1:-}"
if [[ -n "${OUTPUT}" ]]; then
  racpith_py "${RACPITH_PROJECT_ROOT}/scripts/00_check_environment.py" \
    --require-conda-env "${RACPITH_CONDA_ENV}" \
    --output "${OUTPUT}"
else
  racpith_py "${RACPITH_PROJECT_ROOT}/scripts/00_check_environment.py" \
    --require-conda-env "${RACPITH_CONDA_ENV}"
fi
