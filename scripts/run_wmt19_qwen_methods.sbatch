#!/bin/bash
# =====================================================================
# All Qwen-based methods on full WMT19 (1997 sentences) for fair
# comparison with the existing NLLB supporting line.
#
#   0  Qwen direct              wait-k=5  K=0
#   1  Qwen DD + JS gate        wait-k=5  K=4  gate_tau=0.15
#   2  Semantic LCP             wait-k=5  K=4
#
# Usage:
#   sbatch --array=0-2 scripts/run_wmt19_qwen_methods.sbatch
# =====================================================================

#SBATCH --job-name=wmt19_qwen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=haolingp@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/wmt19_qwen_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/wmt19_qwen_%A_%a.err

source ~/.bashrc
export HF_HOME="/data/user_data/haolingp/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
set -euo pipefail

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
OUT_ROOT="${REPO_ROOT}/outputs/wmt19_qwen"

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
VLLM_PORT=$((8240 + SLURM_ARRAY_TASK_ID))
VLLM_API="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="qwen30b-instruct"

# ── Experiment matrix ────────────────────────────────────────────────
declare -a TYPES=(direct_translate  qwen_dd_gate  semantic_lcp)
declare -a KS=(0  4  4)
declare -a TAUS=(0.00  0.15  0.00)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
EXP_TYPE="${TYPES[$IDX]}"
K="${KS[$IDX]}"
TAU="${TAUS[$IDX]}"

case "${EXP_TYPE}" in
    direct_translate) OUT_DIR="${OUT_ROOT}/qwen_direct_k${WAIT_K}" ;;
    qwen_dd_gate)     OUT_DIR="${OUT_ROOT}/qwen_dd_k${WAIT_K}_f${K}_tau${TAU}" ;;
    semantic_lcp)     OUT_DIR="${OUT_ROOT}/semlcp_k${WAIT_K}_f${K}" ;;
esac

# Skip if already done
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] Already complete: ${OUT_DIR}"
    cat "${OUT_DIR}/scores"
    exit 0
fi

mkdir -p "${OUT_DIR}" "${REPO_ROOT}/outputs/slurm_logs"

echo "===== WMT19 QWEN TASK ${IDX}: ${EXP_TYPE} ====="
echo "  wait-k=${WAIT_K}  futures=${K}  tau=${TAU}"
echo "  output=${OUT_DIR}"
echo "  data=${SRC} ($(wc -l < "${SRC}") sentences)"

# ── Start vLLM server ────────────────────────────────────────────────
echo "[vLLM] Starting on GPU ${VLLM_GPU}, port ${VLLM_PORT}"
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

# ── Build agent args ─────────────────────────────────────────────────
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

# ── Run SimulEval ────────────────────────────────────────────────────
echo "[RUN] ${EXP_TYPE} wait-k=${WAIT_K} futures=${K} tau=${TAU}"
CUDA_VISIBLE_DEVICES="${AGENT_GPU}" "${SIMULEVAL}" \
    "${AGENT_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo "===== DONE TASK ${IDX} ====="
kill "${VLLM_PID}" 2>/dev/null || true
