#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "${MPGNN_PYTHON:-}" && -x "${MPGNN_PYTHON}" ]]; then
  PYTHON="${MPGNN_PYTHON}"
elif [[ -x "${CONDA_PREFIX:-}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "No Python interpreter found" >&2
  exit 1
fi

cd "${ROOT}"
"${PYTHON}" -m py_compile \
  data/generate_offline_data.py \
  data/build_offline_from_seed_sets.py
PYTHONPATH="${ROOT}/MPGNN" "${PYTHON}" -m unittest -q \
  MPGNN/tests/test_model_ablation.py \
  MPGNN/tests/test_influence_evaluator.py \
  MPGNN/tests/test_partition_inference.py \
  MPGNN/tests/test_statistical_analysis.py
