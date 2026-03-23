#!/usr/bin/env bash
# =============================================================================
# run_qwen30b.sbatch
# Qwen3-30B-A3B-Instruct (FP8 MoE) continuation — 3-task array
#
# Array layout:
#   0  Qwen30B-Instruct continuation  wait-k=5   (same latency as baseline)
#   1  Qwen30B-Instruct continuation  wait-k=7   (more context before first commit)
#   2  Qwen30B-Instruct continuation  wait-k=9   (even more context)
#
# Why multiple wait-k values:
#   Larger wait-k = more source seen before first commit
#   = fewer early-commit errors at the cost of higher latency
#   This lets us trace the quality-latency Pareto front for Qwen30B.
#
# Note: Qwen3-30B-A3B is a MoE model — only ~3B params are active per token.
#       FP8 quantization keeps memory at ~30GB. Fits on one L40S (46GB).
#
# After all tasks finish:
#   python3 scripts/compare_three_methods.py
#   python3 scripts/compare_waitk_dd.py
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_qwen30b.sbatch
# =============================================================================
#SBATCH --job-name=qwen30b_cont
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-2
#SBATCH --time=0-06:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/qwen30b_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/qwen30b_%A_%a.err

set -e

echo "===== START QWEN30B TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID}  node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

source ~/.bashrc
conda activate base 2>/dev/null || true
export HF_HOME="/data/user_data/haolingp/hf_cache"
CUDA_DEV=0

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
cd "${REPO_ROOT}"

SRC="data/enzh/rand100_source.txt"
TGT="data/enzh/rand100_target.txt"
QWEN30B="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

# ── Per-task wait-k config ────────────────────────────────────────────────────
declare -a WAIT_KS=( 5    7    9  )
T=${SLURM_ARRAY_TASK_ID}
WAIT_K="${WAIT_KS[$T]}"
OUT_DIR="${REPO_ROOT}/outputs/cmp_qwen30b_cont_k${WAIT_K}"

mkdir -p "${REPO_ROOT}/outputs/slurm_logs"
mkdir -p "${OUT_DIR}"

echo "──────────────────────────────────────────────────────"
echo "Experiment  : Qwen30B-Instruct continuation  wait-k=${WAIT_K}"
echo "Model       : ${QWEN30B}"
echo "Output      : ${OUT_DIR}"
echo "──────────────────────────────────────────────────────"

# ── Skip if already complete ───────────────────────────────────────────────────
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] scores already exists. Delete ${OUT_DIR} to re-run."
    cat "${OUT_DIR}/scores"
    echo "===== DONE (skipped) TASK ${SLURM_ARRAY_TASK_ID} ====="
    exit 0
fi

if [[ -d "${OUT_DIR}" ]] && [[ ! -f "${OUT_DIR}/scores" ]]; then
    echo "[Cleanup] Removing partial result at ${OUT_DIR}"
    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"
fi

echo "[RUN] Qwen30B-Instruct continuation  wait-k=${WAIT_K}"
echo ""

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" simuleval \
    --agent    "${REPO_ROOT}/agents/sttr_enzh_agent.py" \
    --source   "${SRC}" \
    --target   "${TGT}" \
    --source-lang eng_Latn \
    --target-lang zho_Hans \
    --wait-k   "${WAIT_K}" \
    --model-name "${QWEN30B}" \
    --causal-lm \
    --causal-instruct \
    --continuation \
    --device   "cuda:${CUDA_DEV}" \
    --output   "${OUT_DIR}" \
    --beam-size 1 \
    --uncertainty-threshold 999 \
    2>&1 | tee "${OUT_DIR}/run.log"

echo ""
echo "[DONE] Qwen30B continuation  wait-k=${WAIT_K}"
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "Scores:"
    cat "${OUT_DIR}/scores"
fi
echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
