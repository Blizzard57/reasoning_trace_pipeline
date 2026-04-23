#!/usr/bin/env bash

set -euo pipefail

# Small local model sweep for reasoning-loop study.
# Usage:
#   bash run_small_models.sh prompts.json
# Optional env overrides:
#   BACKEND=mlx JUDGE_BACKEND=mlx JUDGE_MODEL=Qwen/Qwen2.5-1.5B-Instruct MAX_NEW_TOKENS=512 bash run_small_models.sh prompts.json

PROMPTS_FILE="${1:-prompts.json}"
if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Prompts file not found: ${PROMPTS_FILE}"
  echo "Usage: bash run_small_models.sh <prompts.json>"
  exit 1
fi

BACKEND="${BACKEND:-mlx}"
JUDGE_BACKEND="${JUDGE_BACKEND:-$BACKEND}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-700}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/model_sweep_small}"

mkdir -p "${OUTPUT_ROOT}"

PRIMARY_MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

echo "Prompt file: ${PROMPTS_FILE}"
echo "Primary backend: ${BACKEND}"
echo "Judge backend: ${JUDGE_BACKEND}"
echo "Judge model: ${JUDGE_MODEL}"
echo "Output root: ${OUTPUT_ROOT}"
echo

for PRIMARY_MODEL in "${PRIMARY_MODELS[@]}"; do
  SAFE_NAME="$(echo "${PRIMARY_MODEL}" | tr '/:' '__')"
  OUT_DIR="${OUTPUT_ROOT}/${SAFE_NAME}"
  mkdir -p "${OUT_DIR}"

  echo "============================================================"
  echo "Running model: ${PRIMARY_MODEL}"
  echo "Output dir: ${OUT_DIR}"
  echo "============================================================"

  python experiments.py \
    --prompts-file "${PROMPTS_FILE}" \
    --segmenters hybrid,graph \
    --run-modes baseline,mitigated \
    --primary-model "${PRIMARY_MODEL}" \
    --judge-model "${JUDGE_MODEL}" \
    --backend "${BACKEND}" \
    --judge-backend "${JUDGE_BACKEND}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --enable-diagnostics \
    --enable-latent \
    --enable-pruning \
    --enable-first-step-filter \
    --enable-ccd \
    --output-dir "${OUT_DIR}"

  python report.py \
    --summary-json "${OUT_DIR}/summary.json" \
    --output-path "${OUT_DIR}/report.md"
done

echo
echo "Done. Review per-model reports under: ${OUTPUT_ROOT}"
