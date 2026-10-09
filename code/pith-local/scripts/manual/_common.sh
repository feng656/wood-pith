#!/usr/bin/env bash
set -euo pipefail

RACPITH_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RACPITH_CONDA_ENV="${RACPITH_CONDA_ENV:-py311}"
RACPITH_CONDA_BIN="${RACPITH_CONDA_BIN:-conda}"
RACPITH_CONFIG="${RACPITH_CONFIG:-${RACPITH_PROJECT_ROOT}/configs/racpith_v1.json}"
export PYTHONPATH="${RACPITH_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Local mask smoke runs may use the current Python interpreter explicitly. The
# production manual workflow remains pinned to conda py311 unless overridden.
if [[ -z "${RACPITH_PYTHON_MODE:-}" ]]; then
  RACPITH_PYTHON_MODE="conda"
fi

racpith_py() {
  if [[ "${RACPITH_PYTHON_MODE}" == "local" ]]; then
    python "$@"
  else
    "${RACPITH_CONDA_BIN}" run --no-capture-output -n "${RACPITH_CONDA_ENV}" python "$@"
  fi
}

racpith_runtime_path() {
  local config_path="$1"
  local field="$2"
  racpith_py "${RACPITH_PROJECT_ROOT}/scripts/00_resolve_runtime_paths.py" \
    --config "${config_path}" \
    --field "${field}"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file does not exist: $1" >&2
    exit 2
  fi
}

require_directory() {
  if [[ ! -d "$1" ]]; then
    echo "required directory does not exist: $1" >&2
    exit 2
  fi
}
