#!/bin/bash
# =====================================================================
# Token Consensus ablation on rand100 (100 sentences).
#
# Ablation axes:
#   A) num-futures K:  4, 6, 10 (baseline), 15   (top_logprobs=10 fixed)
#   B) top-logprobs:   5, 10 (baseline), 20       (K=10 fixed)
#
# Deduped grid (K=10, top=10 shared):
#   0  K=4   top=10
#   1  K=6   top=10
#   2  K=10  top=10   ← baseline (skip if fair_consensus_k5 exists)
#   3  K=15  top=10
#   4  K=10  top=5
#   5  K=10  top=20
#
# Usage:
#   sbatch --array=0-5 scripts/run_tc_ablation.sbatch
# =====================================================================

#SBATCH --job-name=tc_ablation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=haolingp@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/tc_ablation_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/tc_ablation_%A_%a.err

source ~/.bashrc
export HF_HOME="/data/user_data/haolingp/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
set -euo pipefail

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
OUT_ROOT="${REPO_ROOT}/outputs/tc_ablation"

CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"
VLLM_PYTHON="${CONDA_BASE}/envs/vllm/bin/python"

# rand100 (100 sentences)
SRC="${REPO_ROOT}/data/enzh/rand100_source.txt"
TGT="${REPO_ROOT}/data/enzh/rand100_target.txt"

QWEN4B_PATH="/data/user_data/haolingp/models/Qwen3-4B-Base"
QWEN30B_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

WAIT_K=5
VLLM_GPU=1
AGENT_GPU=0
VLLM_PORT=$((8260 + SLURM_ARRAY_TASK_ID))
VLLM_API="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="qwen30b-instruct"

# ── Ablation grid ────────────────────────────────────────────────────
declare -a NUM_FUTURES=(4   6   10  15  10  10)
declare -a TOP_LOGPS=(  10  10  10  10  5   20)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
K="${NUM_FUTURES[$IDX]}"
TOP="${TOP_LOGPS[$IDX]}"

OUT_DIR="${OUT_ROOT}/tc_k${WAIT_K}_K${K}_top${TOP}"

# Skip if already done
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] Already complete: ${OUT_DIR}"
    cat "${OUT_DIR}/scores"
    exit 0
fi

mkdir -p "${OUT_DIR}" "${REPO_ROOT}/outputs/slurm_logs"

echo "===== TC ABLATION TASK ${IDX} ====="
echo "  K=${K}  top_logprobs=${TOP}  wait-k=${WAIT_K}"
echo "  output=${OUT_DIR}"

# ── Start vLLM server ────────────────────────────────────────────────
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
echo "[RUN] Token Consensus K=${K} top_logprobs=${TOP}"
CUDA_VISIBLE_DEVICES="${AGENT_GPU}" "${SIMULEVAL}" \
    --agent "${REPO_ROOT}/agents/token_consensus_agent.py" \
    --source "${SRC}" \
    --target "${TGT}" \
    --wait-k "${WAIT_K}" \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --vllm-tokenizer-path "${QWEN30B_PATH}" \
    --num-futures "${K}" \
    --top-logprobs "${TOP}" \
    --max-consensus-steps 6 \
    --base-gpu 0 \
    --output "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo "===== DONE TC ABLATION TASK ${IDX} ====="
kill "${VLLM_PID}" 2>/dev/null || true
