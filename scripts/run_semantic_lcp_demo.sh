#!/usr/bin/env bash
# Quick Semantic LCP smoke run (5 sentences). Requires a running vLLM server for Qwen3-30B.
#
# 1) Start vLLM (OpenAI-compatible), e.g. on port 8100, model name matching --vllm-model-name.
# 2) From repo root:
#      bash scripts/run_semantic_lcp_demo.sh
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUT="${REPO_ROOT}/outputs/demo_semantic_lcp"
mkdir -p "${OUT}"

: "${VLLM_API:=http://localhost:8100/v1}"
: "${VLLM_MODEL_NAME:=qwen30b-instruct}"
: "${QWEN4B_PATH:=/data/user_data/haolingp/models/Qwen3-4B-Base}"

echo "[demo] vLLM base: ${VLLM_API}  model: ${VLLM_MODEL_NAME}"
echo "[demo] Future LM: ${QWEN4B_PATH}"
echo "[demo] Output: ${OUT}  (scores, instances.log, lcp_trace.jsonl when using --output)"

simuleval \
  --agent "${REPO_ROOT}/agents/semantic_lcp_agent.py" \
  --source "${REPO_ROOT}/data/enzh/test_source_5.txt" \
  --target "${REPO_ROOT}/data/enzh/test_target_5.txt" \
  --wait-k 5 \
  --num-futures 4 \
  --verbose \
  --future-lm-path "${QWEN4B_PATH}" \
  --vllm-api-base "${VLLM_API}" \
  --vllm-model-name "${VLLM_MODEL_NAME}" \
  --output "${OUT}"

echo "[demo] Done. Verbose trace JSONL: ${OUT}/lcp_trace.jsonl"
