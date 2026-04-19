#!/usr/bin/env bash
# NLLB STTR with DD full gate. English futures come from --dd-future-lm only (no oracle).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
OUT="${REPO_ROOT}/outputs/demo_dd_full"
mkdir -p "${OUT}"

: "${QWEN4B_PATH:=/data/user_data/haolingp/models/Qwen3-4B-Base}"
echo "[demo] DD full gate -> ${OUT}  (dd_trace.jsonl)  future LM: ${QWEN4B_PATH}"

simuleval \
  --agent "${REPO_ROOT}/agents/sttr_enzh_agent.py" \
  --source "${REPO_ROOT}/data/enzh/test_source_5.txt" \
  --target "${REPO_ROOT}/data/enzh/test_target_5.txt" \
  --wait-k 5 \
  --uncertainty-threshold 1.5 \
  --dd-gate \
  --dd-tau 0.05 \
  --dd-futures-k 4 \
  --dd-steps 3 \
  --dd-future-lm "${QWEN4B_PATH}" \
  --dd-future-words 15 \
  --dd-future-temperature 0.9 \
  --output "${OUT}"

echo "[demo] Done. Inspect ${OUT}/dd_trace.jsonl"
