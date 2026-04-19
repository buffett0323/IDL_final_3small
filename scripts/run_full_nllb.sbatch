#!/usr/bin/env bash
# =============================================================================
# run_full_nllb.sbatch
# Full-scale NLLB experiments on WMT19 EN→ZH (1997 sentences)
#
# Array layout:
#   0  NLLB baseline        wait-k=3
#   1  NLLB baseline        wait-k=5
#   2  NLLB baseline        wait-k=7
#   3  NLLB DD full gate    wait-k=5  tau=0.05
#   4  NLLB DD veto         wait-k=5  tau=0.03
#   5  NLLB LM-sampled DD   wait-k=5  tau=0.03  (Qwen3-4B futures)
#
# All outputs → outputs/full/
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_full_nllb.sbatch
# =============================================================================
#SBATCH --job-name=full_nllb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-5
#SBATCH --time=0-16:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/full_nllb_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/full_nllb_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

echo "===== FULL NLLB TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID}  node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate base 2>/dev/null || true

export HF_HOME="/data/user_data/haolingp/hf_cache"
CUDA_DEV=0

CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
MODEL_ROOT="/data/user_data/haolingp/models"
cd "${REPO_ROOT}"

SRC="data/enzh/wmt19_source.txt"     # 1997 sentences
TGT="data/enzh/wmt19_target.txt"
NLLB_MODEL="facebook/nllb-200-distilled-600M"
QWEN4B_PATH="${MODEL_ROOT}/Qwen3-4B-Base"

# DD hyper-parameters (fixed across all DD experiments)
DD_K=4
DD_STEPS=3

# ── Per-task config ───────────────────────────────────────────────────────────
declare -a TYPES=(   baseline  baseline  baseline  dd_full   dd_veto   lm_dd  )
declare -a WAIT_KS=( 3         5         7         5         5         5      )
declare -a TAUS=(    0.00      0.00      0.00      0.05      0.03      0.03   )

T=${SLURM_ARRAY_TASK_ID}
EXP_TYPE="${TYPES[$T]}"
WAIT_K="${WAIT_KS[$T]}"
TAU="${TAUS[$T]}"

# ── Output directory ──────────────────────────────────────────────────────────
mkdir -p "${REPO_ROOT}/outputs/full"
mkdir -p "${REPO_ROOT}/outputs/slurm_logs"

case "${EXP_TYPE}" in
    baseline) OUT_DIR="${REPO_ROOT}/outputs/full/nllb_baseline_k${WAIT_K}" ;;
    dd_full)  OUT_DIR="${REPO_ROOT}/outputs/full/nllb_dd_full_tau${TAU}"   ;;
    dd_veto)  OUT_DIR="${REPO_ROOT}/outputs/full/nllb_dd_veto_tau${TAU}"   ;;
    lm_dd)    OUT_DIR="${REPO_ROOT}/outputs/full/nllb_lmsdd_tau${TAU}"     ;;
esac

mkdir -p "${OUT_DIR}"

echo "──────────────────────────────────────────────────────"
echo "Experiment : ${EXP_TYPE}  wait-k=${WAIT_K}  tau=${TAU}"
echo "Data       : ${SRC} (1997 sentences)"
echo "Output     : ${OUT_DIR}"
echo "──────────────────────────────────────────────────────"

# ── Skip if already complete ──────────────────────────────────────────────────
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] Already complete. Scores:"
    cat "${OUT_DIR}/scores"
    echo "===== DONE (skipped) TASK ${SLURM_ARRAY_TASK_ID} ====="
    exit 0
fi

# Remove any partial run
if [[ -d "${OUT_DIR}" ]] && [[ ! -f "${OUT_DIR}/scores" ]]; then
    echo "[Cleanup] Removing partial result at ${OUT_DIR}"
    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"
fi

# ── Build command ─────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --agent       "${REPO_ROOT}/agents/sttr_enzh_agent.py"
    --source      "${SRC}"
    --target      "${TGT}"
    --source-lang eng_Latn
    --target-lang zho_Hans
    --wait-k      "${WAIT_K}"
    --model-name  "${NLLB_MODEL}"
    --device      "cuda:${CUDA_DEV}"
    --output      "${OUT_DIR}"
    --beam-size   1
)

case "${EXP_TYPE}" in
    baseline)
        EXTRA_ARGS=(--uncertainty-threshold 999)
        ;;
    dd_full)
        EXTRA_ARGS=(
            --uncertainty-threshold 999
            --dd-gate
            --dd-tau       "${TAU}"
            --dd-futures-k "${DD_K}"
            --dd-steps     "${DD_STEPS}"
            --dd-future-lm "${QWEN4B_PATH}"
            --dd-future-words 15
            --dd-future-temperature 0.9
        )
        ;;
    dd_veto)
        EXTRA_ARGS=(
            --uncertainty-threshold 3.0
            --uncertainty-mode tail3
            --dd-veto
            --dd-tau       "${TAU}"
            --dd-futures-k "${DD_K}"
            --dd-steps     "${DD_STEPS}"
            --dd-future-lm "${QWEN4B_PATH}"
            --dd-future-words 15
            --dd-future-temperature 0.9
        )
        ;;
    lm_dd)
        # DD full gate + LM-sampled English futures (same as dd_full; kept for output dir naming).
        EXTRA_ARGS=(
            --uncertainty-threshold 999
            --dd-gate
            --dd-tau        "${TAU}"
            --dd-futures-k  "${DD_K}"
            --dd-steps      "${DD_STEPS}"
            --dd-future-lm    "${QWEN4B_PATH}"
            --dd-future-words 15
            --dd-future-temperature 0.9
        )
        ;;
esac

echo "[RUN] simuleval ${COMMON_ARGS[*]} ${EXTRA_ARGS[*]}"
echo ""

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" "${SIMULEVAL}" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo ""
echo "[DONE] ${EXP_TYPE}  wait-k=${WAIT_K}  tau=${TAU}"
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "=== SCORES ==="
    cat "${OUT_DIR}/scores"
fi
echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
