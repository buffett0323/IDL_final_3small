#!/usr/bin/env bash
# =============================================================================
# run_continuation.sbatch
# Prefix-constrained continuation (NLLB) vs re-translation baseline — 2 GPUs
#
# Array layout (SLURM_ARRAY_TASK_ID → experiment):
#   0  baseline      wait-k=5  (re-translation, take t-th char)
#   1  continuation  wait-k=5  (forced decoder prefix, generate next char)
#
# Both use the same model (NLLB-200-distilled-600M), same data (rand100),
# same wait-k=5.  The ONLY difference is how the next Chinese character is
# determined at each step.
#
# After both tasks finish, run the comparison script:
#   python3 scripts/compare_continuation.py
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_continuation.sbatch
# =============================================================================
#SBATCH --job-name=cont_vs_retrans
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --time=0-06:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/cont_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/cont_%A_%a.err

set -e

echo "===== START CONTINUATION TASK ${SLURM_ARRAY_TASK_ID} ====="
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
MODEL="facebook/nllb-200-distilled-600M"

# ── Per-task config ───────────────────────────────────────────────────────────
declare -a TYPES=(   baseline     continuation )
declare -a OUT_DIRS=( "outputs/cmp_baseline_k5"   "outputs/cmp_continuation_k5" )

T=${SLURM_ARRAY_TASK_ID}
EXP_TYPE="${TYPES[$T]}"
OUT_DIR="${REPO_ROOT}/${OUT_DIRS[$T]}"

mkdir -p "${REPO_ROOT}/outputs/slurm_logs"
mkdir -p "${OUT_DIR}"

echo "──────────────────────────────────────────────────────"
echo "Experiment  : ${EXP_TYPE}"
echo "Output      : ${OUT_DIR}"
echo "──────────────────────────────────────────────────────"

# ── Skip if already complete ──────────────────────────────────────────────────
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] ${OUT_DIR}/scores already exists.  Delete to re-run."
    cat "${OUT_DIR}/scores"
    echo "===== DONE (skipped) TASK ${SLURM_ARRAY_TASK_ID} ====="
    exit 0
fi

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
    --wait-k   5
    --model-name "${MODEL}"
    --device   "cuda:${CUDA_DEV}"
    --output   "${OUT_DIR}"
    --beam-size 1
    --uncertainty-threshold 999
)

case "${EXP_TYPE}" in
    baseline)
        # Pure wait-k: re-translate full source prefix at each step,
        # emit character at position tgt_len from the draft translation.
        EXTRA_ARGS=()
        ;;
    continuation)
        # Prefix-constrained continuation: force decoder with committed prefix,
        # generate only the next character rather than re-translating entirely.
        EXTRA_ARGS=(
            --continuation
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
echo "[DONE] ${EXP_TYPE}"
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "Scores:"
    cat "${OUT_DIR}/scores"
fi
echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
