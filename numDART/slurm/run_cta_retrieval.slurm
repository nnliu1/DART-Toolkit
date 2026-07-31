#!/bin/bash
#SBATCH --job-name=numdart-cta
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

: "${NUMDART_REPO:?Set NUMDART_REPO to the DART-Toolkit checkout}"
: "${DART_REPO:?Set DART_REPO to the nnliu1/DART checkout}"
: "${DART_MODEL_PATH:?Set DART_MODEL_PATH to the trained checkpoint}"
: "${RUN_DIR:?Set RUN_DIR to the prepared output directory}"
: "${CONFIG_PATH:?Set CONFIG_PATH to the retrieval configuration}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the DART environment Python}"

"${PYTHON_BIN}" "${DART_REPO}/src/cta/run_retrieve.py" \
  --model_path "${DART_MODEL_PATH}" \
  --ontology_path "${RUN_DIR}/ontology.json" \
  --data_dir "${RUN_DIR}/columns" \
  --output_pkl "${RUN_DIR}/retrieval.pkl" \
  --top_k "${TOP_K:-20}" \
  --max_cells "${MAX_CELLS:-10}" \
  --max_parents "${MAX_PARENTS:-3}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --device cuda

cd "${NUMDART_REPO}"
"${PYTHON_BIN}" -m numDART.retrieval.cli export --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m numDART.retrieval.cli validate --config "${CONFIG_PATH}"
