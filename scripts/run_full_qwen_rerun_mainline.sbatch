#!/usr/bin/env bash
# =============================================================================
# run_full_qwen_rerun_mainline.sbatch
# Re-run the mainline fair comparison:
#   Qwen30B direct vs Semantic LCP
#
# Outputs go to:
#   outputs/rerun_20260330/qwen_main/
# =============================================================================
#SBATCH --job-name=qwen_rerun_main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --time=0-24:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/qwen_rerun_main_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/qwen_rerun_main_%A_%a.err

set -e

source ~/.bashrc
export HF_HOME="/data/user_data/haolingp/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

CONDA_BASE="/home/haolingp/miniconda3"
VLLM_PYTHON="${CONDA_BASE}/envs/vllm/bin/python"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
MODEL_ROOT="/data/user_data/haolingp/models"
OUT_ROOT="${REPO_ROOT}/outputs/rerun_20260330/qwen_main"
cd "${REPO_ROOT}"

SRC="data/enzh/wmt500_source.txt"
TGT="data/enzh/wmt500_target.txt"
QWEN30B_PATH="${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507-FP8"
QWEN4B_PATH="${MODEL_ROOT}/Qwen3-4B-Base"

VLLM_PORT=8211
VLLM_API="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="qwen30b"
VLLM_GPU=0
AGENT_GPU=1

declare -a TYPES=(   direct_translate   semantic_lcp )
declare -a WAIT_KS=( 5                 5            )
declare -a NUM_K=(   0                 4            )

T=${SLURM_ARRAY_TASK_ID}
EXP_TYPE="${TYPES[$T]}"
WAIT_K="${WAIT_KS[$T]}"
K="${NUM_K[$T]}"

mkdir -p "${OUT_ROOT}" "${REPO_ROOT}/outputs/slurm_logs"

case "${EXP_TYPE}" in
    direct_translate) OUT_DIR="${OUT_ROOT}/qwen_direct_k${WAIT_K}" ;;
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

echo "[RUN] ${EXP_TYPE} wait-k=${WAIT_K} futures=${K}"
CUDA_VISIBLE_DEVICES="${AGENT_GPU}" "${SIMULEVAL}" \
    "${AGENT_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"

kill "${VLLM_PID}" 2>/dev/null || true
