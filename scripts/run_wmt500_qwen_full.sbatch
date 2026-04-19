#!/usr/bin/env bash
# =============================================================================
# run_wmt500_qwen_full.sbatch
# Qwen experiments on wmt500 (500 sentences) — for fair two-group comparison
#
# Architecture (per task):
#   GPU 0  → vLLM serve Qwen3-30B-A3B-Instruct-2507-FP8
#   GPU 1  → Qwen3-4B-Base future LM + SimulEval agent
#
# Array layout:
#   0  Qwen30B direct   wait-k=3   K=0   (low-latency baseline)
#   1  Qwen30B direct   wait-k=7   K=0   (high-latency baseline)
#   2  Future LCP       wait-k=3   K=4   (low-latency Method 2)
#   3  Future LCP       wait-k=7   K=4   (high-latency Method 2)
#   4  Future LCP       wait-k=5   K=6   (Method 2 more futures)
#
# Already done (skip automatically):
#   Qwen30B direct k=5 → outputs/full/qwen_direct_k5
#   Future LCP     k=5 K=4 → outputs/full/semlcp_k5_f4
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_wmt500_qwen_full.sbatch
# =============================================================================
#SBATCH --job-name=wmt500_qwen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-4
#SBATCH --time=0-24:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/wmt500_qwen_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/wmt500_qwen_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

echo "===== WMT500 QWEN TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID}  node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
export HF_HOME="/data/user_data/haolingp/hf_cache"

CONDA_BASE="/home/haolingp/miniconda3"
VLLM_PYTHON="${CONDA_BASE}/envs/vllm/bin/python"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
MODEL_ROOT="/data/user_data/haolingp/models"
cd "${REPO_ROOT}"

SRC="data/enzh/wmt500_source.txt"
TGT="data/enzh/wmt500_target.txt"

QWEN30B_PATH="${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507-FP8"
QWEN4B_PATH="${MODEL_ROOT}/Qwen3-4B-Base"

VLLM_PORT=8202
VLLM_API="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="qwen30b"

VLLM_GPU=0
AGENT_GPU=1

# ── Per-task config ───────────────────────────────────────────────────────────
declare -a TYPES=(   direct_translate  direct_translate  semantic_lcp  semantic_lcp  semantic_lcp )
declare -a WAIT_KS=( 3                 7                 3             7             5            )
declare -a NUM_K=(   0                 0                 4             4             6            )

T=${SLURM_ARRAY_TASK_ID}
EXP_TYPE="${TYPES[$T]}"
WAIT_K="${WAIT_KS[$T]}"
K="${NUM_K[$T]}"

# ── Output directory ──────────────────────────────────────────────────────────
mkdir -p "${REPO_ROOT}/outputs/full"
mkdir -p "${REPO_ROOT}/outputs/slurm_logs"

case "${EXP_TYPE}" in
    direct_translate) OUT_DIR="${REPO_ROOT}/outputs/full/qwen_direct_k${WAIT_K}" ;;
    semantic_lcp)     OUT_DIR="${REPO_ROOT}/outputs/full/semlcp_k${WAIT_K}_f${K}" ;;
esac

mkdir -p "${OUT_DIR}"

echo "──────────────────────────────────────────────────────"
echo "Experiment : ${EXP_TYPE}  wait-k=${WAIT_K}  futures=${K}"
echo "Data       : ${SRC} (500 sentences)"
echo "Output     : ${OUT_DIR}"
echo "──────────────────────────────────────────────────────"

# ── Skip if already complete ──────────────────────────────────────────────────
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] Already complete:"
    cat "${OUT_DIR}/scores"
    echo "===== DONE (skipped) TASK ${SLURM_ARRAY_TASK_ID} ====="
    exit 0
fi

if [[ -d "${OUT_DIR}" ]] && [[ ! -f "${OUT_DIR}/scores" ]]; then
    echo "[Cleanup] Removing partial result at ${OUT_DIR}"
    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"
fi

# ── Start vLLM server ─────────────────────────────────────────────────────────
echo "[vLLM] Starting Qwen3-30B-A3B-Instruct-2507-FP8 on GPU ${VLLM_GPU}, port ${VLLM_PORT} ..."
VLLM_LOG="${OUT_DIR}/vllm_server.log"

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model              "${QWEN30B_PATH}" \
    --served-model-name  "${VLLM_MODEL_NAME}" \
    --port               "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --dtype              float16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len      4096 \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[vLLM] Server PID: ${VLLM_PID}"

# Poll until server is ready (up to 5 minutes)
MAX_WAIT=300
WAITED=0
while true; do
    if curl -s "${VLLM_API}/models" > /dev/null 2>&1; then
        echo "[vLLM] Ready after ${WAITED}s."
        break
    fi
    sleep 5; WAITED=$((WAITED + 5))
    if [[ ${WAITED} -ge ${MAX_WAIT} ]]; then
        echo "[ERROR] vLLM did not start within ${MAX_WAIT}s. Log:"
        tail -40 "${VLLM_LOG}"
        kill "${VLLM_PID}" 2>/dev/null || true
        exit 1
    fi
done

# ── Run agent ─────────────────────────────────────────────────────────────────
AGENT_ARGS=(
    --agent              "${REPO_ROOT}/agents/semantic_lcp_agent.py"
    --source             "${SRC}"
    --target             "${TGT}"
    --wait-k             "${WAIT_K}"
    --future-lm-path     "${QWEN4B_PATH}"
    --vllm-api-base      "${VLLM_API}"
    --vllm-model-name    "${VLLM_MODEL_NAME}"
    --num-futures        "${K}"
    --future-words       15
    --future-temperature 0.9
    --consensus-ratio    0.6
    --base-gpu           0
    --output             "${OUT_DIR}"
)

echo "[RUN] ${EXP_TYPE}  wait-k=${WAIT_K}  K=${K}"
CUDA_VISIBLE_DEVICES="${AGENT_GPU}" "${SIMULEVAL}" \
    "${AGENT_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo ""
echo "[DONE] ${EXP_TYPE}"
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "=== SCORES ==="
    cat "${OUT_DIR}/scores"
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "[vLLM] Stopping server (PID ${VLLM_PID})"
kill "${VLLM_PID}" 2>/dev/null || true

echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
