#!/bin/bash
# =====================================================================
# Token Consensus Decoding on full WMT19 (1997 sentences).
#
# Separate from the other Qwen methods because Token Consensus needs
# --return-tokens-as-token-ids on the vLLM server.
#
# Estimated runtime: Token Consensus is ~5-10x slower per sentence
# than SemLCP due to iterative per-token API calls (K=10 futures,
# up to 6 consensus steps per source word). 48h limit is set
# conservatively; adjust if needed.
#
# Usage:
#   sbatch scripts/run_wmt19_token_consensus.sbatch
# =====================================================================

#SBATCH --job-name=wmt19_tc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=haolingp@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/wmt19_tc_%j.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/wmt19_tc_%j.err

source ~/.bashrc
export HF_HOME="/data/user_data/haolingp/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
set -euo pipefail

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
OUT_DIR="${REPO_ROOT}/outputs/wmt19_qwen/token_consensus_k5"

CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"
VLLM_PYTHON="${CONDA_BASE}/envs/vllm/bin/python"

# Full WMT19 (1997 sentences)
SRC="${REPO_ROOT}/data/enzh/wmt19_source.txt"
TGT="${REPO_ROOT}/data/enzh/wmt19_target.txt"

QWEN4B_PATH="/data/user_data/haolingp/models/Qwen3-4B-Base"
QWEN30B_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

WAIT_K=5
VLLM_GPU=1
AGENT_GPU=0
VLLM_PORT=8250
VLLM_API="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="qwen30b-instruct"

# Skip if already done
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] Already complete: ${OUT_DIR}"
    cat "${OUT_DIR}/scores"
    exit 0
fi

mkdir -p "${OUT_DIR}" "${REPO_ROOT}/outputs/slurm_logs"

echo "===== WMT19 TOKEN CONSENSUS ====="
echo "  wait-k=${WAIT_K}  K=10  top_logprobs=10  max_steps=6"
echo "  output=${OUT_DIR}"
echo "  data=${SRC} ($(wc -l < "${SRC}") sentences)"

# ── Start vLLM server (with --return-tokens-as-token-ids) ────────────
echo "[vLLM] Starting on GPU ${VLLM_GPU}, port ${VLLM_PORT}"
VLLM_TARGET_DEVICE=cuda CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${VLLM_PYTHON}" \
    -m vllm.entrypoints.openai.api_server \
    --model "${QWEN30B_PATH}" \
    --served-model-name "${VLLM_MODEL_NAME}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --return-tokens-as-token-ids \
    > "${OUT_DIR}/vllm_server.log" 2>&1 &

VLLM_PID=$!
trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT

MAX_WAIT=300
WAITED=0
while true; do
    if curl -s "${VLLM_API}/models" > /dev/null 2>&1; then
        echo "[vLLM] Ready after ${WAITED}s"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [[ ${WAITED} -ge ${MAX_WAIT} ]]; then
        echo "[vLLM] FAILED to start after ${MAX_WAIT}s"
        tail -40 "${OUT_DIR}/vllm_server.log"
        exit 1
    fi
done

# ── Run SimulEval ────────────────────────────────────────────────────
echo "[RUN] Token Consensus wait-k=${WAIT_K} K=10"
CUDA_VISIBLE_DEVICES="${AGENT_GPU}" "${SIMULEVAL}" \
    --agent "${REPO_ROOT}/agents/token_consensus_agent.py" \
    --source "${SRC}" \
    --target "${TGT}" \
    --wait-k "${WAIT_K}" \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --vllm-tokenizer-path "${QWEN30B_PATH}" \
    --num-futures 10 \
    --top-logprobs 10 \
    --max-consensus-steps 6 \
    --base-gpu 0 \
    --output "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo "===== DONE TOKEN CONSENSUS ====="
kill "${VLLM_PID}" 2>/dev/null || true
