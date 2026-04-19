#!/usr/bin/env bash
# Run Method 3 token consensus on rand100 with a local vLLM server.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUT="${REPO_ROOT}/outputs/token_consensus_rand100"
mkdir -p "${OUT}"

CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"
VLLM_PYTHON="${CONDA_BASE}/envs/vllm/bin/python"

: "${VLLM_PORT:=8236}"
: "${VLLM_API:=http://localhost:${VLLM_PORT}/v1}"
: "${VLLM_MODEL_NAME:=qwen30b-instruct}"
: "${QWEN4B_PATH:=/data/user_data/haolingp/models/Qwen3-4B-Base}"
: "${QWEN30B_PATH:=/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8}"

echo "[rand100] Starting vLLM on ${VLLM_API}"
VLLM_TARGET_DEVICE=cuda CUDA_VISIBLE_DEVICES=1 "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${QWEN30B_PATH}" \
  --served-model-name "${VLLM_MODEL_NAME}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --return-tokens-as-token-ids \
  > "${OUT}/vllm_server.log" 2>&1 &

VLLM_PID=$!
trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT

MAX_WAIT=240
WAITED=0
while true; do
  if curl -s "${VLLM_API}/models" > /dev/null 2>&1; then
    break
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  if [[ ${WAITED} -ge ${MAX_WAIT} ]]; then
    echo "[rand100] vLLM failed to start"
    tail -40 "${OUT}/vllm_server.log" || true
    exit 1
  fi
done

echo "[rand100] Running SimulEval on rand100"
CUDA_VISIBLE_DEVICES=0 "${SIMULEVAL}" \
  --agent "${REPO_ROOT}/agents/token_consensus_agent.py" \
  --source "${REPO_ROOT}/data/enzh/rand100_source.txt" \
  --target "${REPO_ROOT}/data/enzh/rand100_target.txt" \
  --wait-k 5 \
  --future-lm-path "${QWEN4B_PATH}" \
  --vllm-api-base "${VLLM_API}" \
  --vllm-model-name "${VLLM_MODEL_NAME}" \
  --vllm-tokenizer-path "${QWEN30B_PATH}" \
  --num-futures 10 \
  --top-logprobs 10 \
  --max-consensus-steps 6 \
  --base-gpu 0 \
  --verbose \
  --output "${OUT}" \
  2>&1 | tee "${OUT}/run.log"

echo "[rand100] Done"
