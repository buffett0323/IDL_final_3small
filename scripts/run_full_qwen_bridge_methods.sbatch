#!/bin/bash
# Compare three Qwen-side methods on the same translator and same data slice:
#   0  Qwen direct              wait-k=5  K=0
#   1  Qwen + DD-style gate     wait-k=5  K=4  gate_tau=0.15
#   2  Semantic LCP             wait-k=5  K=4
#
# This script is meant to answer a cleaner question for the final project:
# on the SAME strong translator, how much do we gain from
#   (a) future-aware gating
#   (b) future-aware consensus
#
# Usage:
#   sbatch --array=0-2 scripts/run_full_qwen_bridge_methods.sbatch

#SBATCH --job-name=qwen_bridge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=0-24:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/qwen_bridge_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/qwen_bridge_%A_%a.err

source ~/.bashrc
export HF_HOME="/data/user_data/haolingp/hf_cache"
set -euo pipefail

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
OUT_ROOT="${REPO_ROOT}/outputs/rerun_20260330/qwen_bridge"

CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"
VLLM_PYTHON="${CONDA_BASE}/envs/vllm/bin/python"

SRC="${REPO_ROOT}/data/enzh/wmt500_source.txt"
TGT="${REPO_ROOT}/data/enzh/wmt500_target.txt"

QWEN4B_PATH="/data/user_data/haolingp/models/Qwen3-4B-Base"
QWEN30B_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

WAIT_K=5
VLLM_GPU=1
AGENT_GPU=0
VLLM_PORT=$((8221 + SLURM_ARRAY_TASK_ID))
VLLM_API="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="qwen30b-instruct"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

declare -a TYPES=(direct_translate qwen_dd_gate semantic_lcp)
declare -a KS=(0 4 4)
declare -a TAUS=(0.00 0.15 0.00)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
EXP_TYPE="${TYPES[$IDX]}"
K="${KS[$IDX]}"
TAU="${TAUS[$IDX]}"

case "${EXP_TYPE}" in
    direct_translate) OUT_DIR="${OUT_ROOT}/qwen_direct_k${WAIT_K}" ;;
    qwen_dd_gate)     OUT_DIR="${OUT_ROOT}/qwen_dd_k${WAIT_K}_f${K}_tau${TAU}" ;;
    semantic_lcp)     OUT_DIR="${OUT_ROOT}/semlcp_k${WAIT_K}_f${K}" ;;
esac

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${QWEN30B_PATH}" \
    --served-model-name "${VLLM_MODEL_NAME}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    > "${OUT_DIR}/vllm_server.log" 2>&1 &

VLLM_PID=$!
MAX_WAIT=240
WAITED=0
while true; do
    if curl -s "${VLLM_API}/models" > /dev/null 2>&1; then
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [[ ${WAITED} -ge ${MAX_WAIT} ]]; then
        tail -40 "${OUT_DIR}/vllm_server.log"
        kill "${VLLM_PID}" 2>/dev/null || true
        exit 1
    fi
done

AGENT_ARGS=(
    --agent "${REPO_ROOT}/agents/semantic_lcp_agent.py"
    --source "${SRC}"
    --target "${TGT}"
    --wait-k "${WAIT_K}"
    --future-lm-path "${QWEN4B_PATH}"
    --vllm-api-base "${VLLM_API}"
    --vllm-model-name "${VLLM_MODEL_NAME}"
    --num-futures "${K}"
    --future-words 15
    --future-temperature 0.9
    --consensus-ratio 0.6
    --base-gpu 0
    --output "${OUT_DIR}"
)

if [[ "${EXP_TYPE}" == "qwen_dd_gate" ]]; then
    AGENT_ARGS+=(
        --gate-js
        --gate-tau "${TAU}"
        --gate-steps 3
    )
fi

echo "[RUN] ${EXP_TYPE} wait-k=${WAIT_K} futures=${K} tau=${TAU}"
CUDA_VISIBLE_DEVICES="${AGENT_GPU}" "${SIMULEVAL}" \
    "${AGENT_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"

kill "${VLLM_PID}" 2>/dev/null || true
