#!/usr/bin/env bash
# =============================================================================
# run_qwen_continuation.sbatch
# Three-way comparison: NLLB baseline vs NLLB continuation vs Qwen continuation
#
# Array layout (SLURM_ARRAY_TASK_ID → experiment):
#   0  NLLB  baseline      wait-k=5  (re-translation, take t-th char)
#   1  NLLB  continuation  wait-k=5  (forced decoder prefix)
#   2  Qwen3-4B-Base  continuation  wait-k=5  (causal LM, natural prefix)
#
# All tasks skip if the output/scores file already exists.
#
# After all tasks finish, run the analysis:
#   python3 scripts/compare_continuation.py
#   python3 scripts/compare_three_methods.py
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_qwen_continuation.sbatch
# =============================================================================
#SBATCH --job-name=qwen_vs_nllb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-2
#SBATCH --time=0-04:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/qwen_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/qwen_%A_%a.err

set -e

echo "===== START TASK ${SLURM_ARRAY_TASK_ID} ====="
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
NLLB_MODEL="facebook/nllb-200-distilled-600M"
QWEN_MODEL="/data/user_data/haolingp/models/Qwen3-4B-Base"

# ── Per-task config ────────────────────────────────────────────────────────────
#  Task 0: NLLB baseline (re-translation, take t-th char)
#  Task 1: NLLB continuation (forced decoder prefix)
#  Task 2: Qwen3-4B-Base continuation (causal LM, natural prefix)
declare -a NAMES=(     "NLLB-baseline-k5"      "NLLB-continuation-k5"   "Qwen-continuation-k5"  )
declare -a OUT_DIRS=(  "cmp_baseline_k5"        "cmp_continuation_k5"    "cmp_qwen_continuation_k5" )
declare -a MODELS=(    "${NLLB_MODEL}"           "${NLLB_MODEL}"           "${QWEN_MODEL}"          )

T=${SLURM_ARRAY_TASK_ID}
EXP_NAME="${NAMES[$T]}"
OUT_DIR="${REPO_ROOT}/outputs/${OUT_DIRS[$T]}"
MODEL="${MODELS[$T]}"

mkdir -p "${REPO_ROOT}/outputs/slurm_logs"
mkdir -p "${OUT_DIR}"

echo "──────────────────────────────────────────────────────"
echo "Experiment  : ${EXP_NAME}"
echo "Model       : ${MODEL}"
echo "Output      : ${OUT_DIR}"
echo "──────────────────────────────────────────────────────"

# ── Skip if already complete ───────────────────────────────────────────────────
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

# ── Build simuleval command ────────────────────────────────────────────────────
COMMON_ARGS=(
    --agent    "${REPO_ROOT}/agents/sttr_enzh_agent.py"
    --source   "${SRC}"
    --target   "${TGT}"
    --source-lang eng_Latn
    --target-lang zho_Hans
    --wait-k   5
    --device   "cuda:${CUDA_DEV}"
    --output   "${OUT_DIR}"
    --beam-size 1
    --uncertainty-threshold 999
)

case "${T}" in
    0)
        # Pure NLLB wait-k baseline: re-translate full prefix, take char at position t.
        EXTRA_ARGS=(
            --model-name "${MODEL}"
        )
        ;;
    1)
        # NLLB continuation: force decoder to start with committed prefix.
        # Eliminates hypothesis inconsistency but suffers from tokenisation mismatch.
        EXTRA_ARGS=(
            --model-name "${MODEL}"
            --continuation
        )
        ;;
    2)
        # Qwen3-4B-Base continuation: causal LM, appends committed Chinese to prompt.
        # Most natural continuation — no encoder-decoder mismatch.
        EXTRA_ARGS=(
            --model-name "${MODEL}"
            --causal-lm
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
echo "[DONE] ${EXP_NAME}"
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "Scores:"
    cat "${OUT_DIR}/scores"
fi
echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
