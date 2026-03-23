#!/usr/bin/env bash
# =============================================================================
# run_semantic_lcp.sbatch
# Semantic LCP Simultaneous Translation — Qwen30B-Instruct + Qwen3-4B futures
#
# Architecture:
#   GPU 0      : vLLM serve Qwen3-30B-A3B-Instruct-2507-FP8 (30GB FP8, fits on 1×L40S)
#   GPU 1      : Qwen3-4B-Base future LM + SimulEval agent
#
# Pipeline per commit step:
#   1. Qwen3-4B  → K English future continuations
#   2. Qwen30B   → translate observed prefix given each future (K API calls)
#   3. Quorum LCP → find chars ≥60% candidates agree on → COMMIT
#
# Array layout:
#   0  baseline       wait-k=5        (NLLB wait-k, no Qwen)
#   1  semantic_lcp   wait-k=5  K=4  (Qwen30B futures + quorum LCP)
#   2  semantic_lcp   wait-k=5  K=6  (more futures)
#   3  direct_translate wait-k=5 K=0 (Qwen30B direct, no futures, fair baseline)
#   4  direct_translate wait-k=3 K=0 (lower latency direct baseline)
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_semantic_lcp.sbatch
# =============================================================================
#SBATCH --job-name=sem_lcp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=120G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=3-4
#SBATCH --time=0-12:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/sem_lcp_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/sem_lcp_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

echo "===== START SEMANTIC LCP TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID}  node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
export HF_HOME="/data/user_data/haolingp/hf_cache"

# Conda env paths (used explicitly to avoid activation issues in sbatch)
CONDA_BASE="/home/haolingp/miniconda3"
VLLM_PYTHON="${CONDA_BASE}/envs/vllm/bin/python"    # has vllm 0.11.0
BASE_PYTHON="${CONDA_BASE}/bin/python"               # has simuleval + transformers
SIMULEVAL="${CONDA_BASE}/bin/simuleval"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
MODEL_ROOT="/data/user_data/haolingp/models"
cd "${REPO_ROOT}"

SRC="data/enzh/rand100_source.txt"
TGT="data/enzh/rand100_target.txt"

QWEN30B_PATH="${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507-FP8"
QWEN4B_PATH="${MODEL_ROOT}/Qwen3-4B-Base"
NLLB_MODEL="facebook/nllb-200-distilled-600M"

VLLM_PORT=8200
VLLM_API="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="qwen30b-instruct"

# GPUs: 0 → vLLM server, 1 → agent (future LM + simuleval)
VLLM_GPU=0
AGENT_GPU=1

# ── Per-task config ───────────────────────────────────────────────────────────
declare -a TYPES=(   baseline   semantic_lcp   semantic_lcp   direct_translate   direct_translate )
declare -a WAIT_KS=( 5          5              5              5                  3                )
declare -a NUM_K=(   0          4              6              0                  0                )

T=${SLURM_ARRAY_TASK_ID}
EXP_TYPE="${TYPES[$T]}"
WAIT_K="${WAIT_KS[$T]}"
K="${NUM_K[$T]}"

# ── Output directory ──────────────────────────────────────────────────────────
case "${EXP_TYPE}" in
    baseline)         OUT_DIR="${REPO_ROOT}/outputs/semlcp_baseline_k${WAIT_K}" ;;
    semantic_lcp)     OUT_DIR="${REPO_ROOT}/outputs/semlcp_main_k${WAIT_K}_f${K}" ;;
    direct_translate) OUT_DIR="${REPO_ROOT}/outputs/semlcp_direct_k${WAIT_K}" ;;
esac

mkdir -p "${REPO_ROOT}/outputs/slurm_logs"
mkdir -p "${OUT_DIR}"

echo "──────────────────────────────────────────────────────"
echo "Experiment : ${EXP_TYPE}  wait-k=${WAIT_K}  futures=${K}"
echo "Output     : ${OUT_DIR}"
echo "──────────────────────────────────────────────────────"

# ── Skip if already complete ──────────────────────────────────────────────────
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] ${OUT_DIR}/scores already exists. Delete to re-run."
    cat "${OUT_DIR}/scores"
    echo "===== DONE (skipped) TASK ${SLURM_ARRAY_TASK_ID} ====="
    exit 0
fi

if [[ -d "${OUT_DIR}" ]] && [[ ! -f "${OUT_DIR}/scores" ]]; then
    echo "[Cleanup] Removing partial result at ${OUT_DIR}"
    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"
fi

# ── Baseline: just use existing NLLB agent, no vLLM needed ───────────────────
if [[ "${EXP_TYPE}" == "baseline" ]]; then
    echo "[RUN] Baseline (NLLB wait-k=${WAIT_K})"
    CUDA_VISIBLE_DEVICES="${AGENT_GPU}" "${SIMULEVAL}" \
        --agent    "${REPO_ROOT}/agents/sttr_enzh_agent.py" \
        --source   "${SRC}" \
        --target   "${TGT}" \
        --source-lang eng_Latn \
        --target-lang zho_Hans \
        --wait-k   "${WAIT_K}" \
        --model-name "${NLLB_MODEL}" \
        --device   "cuda:0" \
        --output   "${OUT_DIR}" \
        --beam-size 1 \
        --uncertainty-threshold 999 \
        2>&1 | tee "${OUT_DIR}/run.log"

    echo "[DONE] Baseline"
    cat "${OUT_DIR}/scores"
    echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
    exit 0
fi

# ── Semantic LCP: start vLLM server first ─────────────────────────────────────
echo "[vLLM] Starting Qwen30B-Instruct on GPU ${VLLM_GPU}, port ${VLLM_PORT}"
VLLM_LOG="${OUT_DIR}/vllm_server.log"

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${QWEN30B_PATH}" \
    --served-model-name "${VLLM_MODEL_NAME}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[vLLM] Server PID: ${VLLM_PID}"

# Wait for server to be ready (poll up to 3 minutes)
echo "[vLLM] Waiting for server to be ready..."
MAX_WAIT=180
WAITED=0
while true; do
    if curl -s "${VLLM_API}/models" > /dev/null 2>&1; then
        echo "[vLLM] Server is ready after ${WAITED}s."
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [[ ${WAITED} -ge ${MAX_WAIT} ]]; then
        echo "[ERROR] vLLM server failed to start within ${MAX_WAIT}s. Check ${VLLM_LOG}"
        tail -30 "${VLLM_LOG}"
        kill ${VLLM_PID} 2>/dev/null || true
        exit 1
    fi
done

# ── Run semantic LCP / direct-translate agent ─────────────────────────────────
if [[ "${EXP_TYPE}" == "direct_translate" ]]; then
    echo "[RUN] Qwen30B Direct-Translate baseline (no futures) wait-k=${WAIT_K}"
else
    echo "[RUN] SemanticLCP wait-k=${WAIT_K} K=${K}"
fi

CUDA_VISIBLE_DEVICES="${AGENT_GPU}" "${SIMULEVAL}" \
    --agent    "${REPO_ROOT}/agents/semantic_lcp_agent.py" \
    --source   "${SRC}" \
    --target   "${TGT}" \
    --wait-k   "${WAIT_K}" \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base  "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --num-futures     "${K}" \
    --future-words    15 \
    --future-temperature 0.9 \
    --consensus-ratio 0.6 \
    --base-gpu 0 \
    --output   "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo ""
echo "[DONE] ${EXP_TYPE}"
if [[ -f "${OUT_DIR}/scores" ]]; then
    cat "${OUT_DIR}/scores"
fi

# ── Cleanup vLLM server ────────────────────────────────────────────────────────
echo "[vLLM] Stopping server (PID ${VLLM_PID})"
kill ${VLLM_PID} 2>/dev/null || true

echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
