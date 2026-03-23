#!/usr/bin/env bash
# =============================================================================
# run_comparison.sbatch
# Wait-k baseline sweep  vs  DD full gate  vs  DD veto — 8-GPU array job
#
# Array layout (SLURM_ARRAY_TASK_ID → experiment):
#   0  baseline  wait-k=3
#   1  baseline  wait-k=5
#   2  baseline  wait-k=7
#   3  baseline  wait-k=9
#   4  DD full gate  wait-k=5  tau=0.03
#   5  DD full gate  wait-k=5  tau=0.05
#   6  DD veto       wait-k=5  tau=0.03
#   7  DD veto       wait-k=5  tau=0.05
#
# After all tasks finish, run the comparison + plot script:
#   python scripts/compare_waitk_dd.py
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_comparison.sbatch
# =============================================================================
#SBATCH --job-name=waitk_dd_cmp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --time=0-08:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/cmp_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/cmp_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

echo "===== START COMPARISON TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID}  node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate base 2>/dev/null || true

export HF_HOME="/data/user_data/haolingp/hf_cache"
CUDA_DEV=0    # each task gets its own GPU (gres=gpu:L40S:1)

# ── Repo & data paths ─────────────────────────────────────────────────────────
REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
cd "${REPO_ROOT}"

SRC="data/enzh/rand100_source.txt"
TGT="data/enzh/rand100_target.txt"
MODEL="facebook/nllb-200-distilled-600M"

# Shared DD hyper-parameters (keep fixed for fair comparison)
DD_K=4
DD_STEPS=3

# ── Per-task config ───────────────────────────────────────────────────────────
# Encode experiment array as parallel bash arrays.
# Columns: TYPE  WAIT_K  DD_TAU  DD_FLAGS
#   TYPE  = baseline | dd_full | dd_veto
declare -a TYPES=( baseline  baseline  baseline  baseline  dd_full   dd_full   dd_veto   dd_veto  )
declare -a WAIT_KS=(  3         5         7         9         5         5         5         5      )
declare -a TAUS=(     0.00      0.00      0.00      0.00      0.03      0.05      0.03      0.05   )

T=${SLURM_ARRAY_TASK_ID}
EXP_TYPE="${TYPES[$T]}"
WAIT_K="${WAIT_KS[$T]}"
TAU="${TAUS[$T]}"

# ── Output directory ──────────────────────────────────────────────────────────
case "${EXP_TYPE}" in
    baseline) OUT_DIR="${REPO_ROOT}/outputs/cmp_baseline_k${WAIT_K}" ;;
    dd_full)  OUT_DIR="${REPO_ROOT}/outputs/cmp_dd_full_tau${TAU}"   ;;
    dd_veto)  OUT_DIR="${REPO_ROOT}/outputs/cmp_dd_veto_tau${TAU}"   ;;
esac

mkdir -p "${REPO_ROOT}/outputs/slurm_logs"
mkdir -p "${OUT_DIR}"

echo "──────────────────────────────────────────────────────"
echo "Experiment : ${EXP_TYPE}  wait-k=${WAIT_K}  tau=${TAU}"
echo "Output     : ${OUT_DIR}"
echo "──────────────────────────────────────────────────────"

# ── Skip if already complete ──────────────────────────────────────────────────
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] ${OUT_DIR}/scores already exists.  Delete to re-run."
    cat "${OUT_DIR}/scores"
    echo "===== DONE (skipped) TASK ${SLURM_ARRAY_TASK_ID} ====="
    exit 0
fi

# Remove any partial run so simuleval starts clean
if [[ -d "${OUT_DIR}" ]] && [[ ! -f "${OUT_DIR}/scores" ]]; then
    echo "[Cleanup] Removing partial result at ${OUT_DIR}"
    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"
fi

# ── Build simuleval command ───────────────────────────────────────────────────
COMMON_ARGS=(
    --agent    "${REPO_ROOT}/agents/sttr_enzh_agent.py"
    --source   "${SRC}"
    --target   "${TGT}"
    --source-lang eng_Latn
    --target-lang zho_Hans
    --wait-k   "${WAIT_K}"
    --model-name "${MODEL}"
    --device   "cuda:${CUDA_DEV}"
    --output   "${OUT_DIR}"
    --beam-size 1
)

case "${EXP_TYPE}" in
    baseline)
        # Pure wait-k: disable entropy gate (tau=999 → never triggers read-more)
        EXTRA_ARGS=(
            --uncertainty-threshold 999
        )
        ;;
    dd_full)
        # DD full gate: DD is the sole commit/read arbiter (post wait-k)
        # Entropy gate disabled so only DD controls decisions
        EXTRA_ARGS=(
            --uncertainty-threshold 999
            --dd-gate
            --dd-tau    "${TAU}"
            --dd-futures-k "${DD_K}"
            --dd-steps  "${DD_STEPS}"
        )
        ;;
    dd_veto)
        # DD veto: baseline entropy gate (tau=3.0) runs first; DD only intercepts
        # a baseline COMMIT decision when divergence is high.
        EXTRA_ARGS=(
            --uncertainty-threshold 3.0
            --uncertainty-mode tail3
            --dd-veto
            --dd-tau    "${TAU}"
            --dd-futures-k "${DD_K}"
            --dd-steps  "${DD_STEPS}"
        )
        ;;
esac

echo "[RUN] simuleval ${COMMON_ARGS[*]} ${EXTRA_ARGS[*]}"
echo ""

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" simuleval \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo ""
echo "[DONE] ${EXP_TYPE} wait-k=${WAIT_K} tau=${TAU}"
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "Scores:"
    cat "${OUT_DIR}/scores"
fi
echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
