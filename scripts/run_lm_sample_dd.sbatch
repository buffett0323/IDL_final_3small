#!/usr/bin/env bash
# =============================================================================
# run_lm_sample_dd.sbatch
# Ablation: Oracle-future DD  vs  LM-sampled-future DD  (vs baseline)
#
# Key idea: instead of revealing oracle truncation futures, use a small causal
# LM (Qwen3-4B-Base) to generate K plausible English continuations on-the-fly.
# This tests whether the DD gate can work WITHOUT oracle knowledge — i.e.,
# whether it is deployable in real-time streaming translation.
#
# Array layout (SLURM_ARRAY_TASK_ID → experiment):
#   0  baseline            wait-k=5  (pure wait-k, no DD)
#   1  DD oracle           wait-k=5  tau=0.05
#   2  DD lm_sample        wait-k=5  tau=0.05  future-lm=Qwen3-4B
#   3  DD lm_sample        wait-k=5  tau=0.03  future-lm=Qwen3-4B
#   4  DD oracle           wait-k=5  tau=0.03  (fair latency comparison vs task 3)
#
# Two models loaded per task for DD lm_sample jobs:
#   - NLLB-200-distilled-600M  (base MT model, ~2.4 GB fp16)
#   - Qwen3-4B-Base             (future LM,    ~8  GB fp16)
#   Total: ~10-11 GB on one L40S (46 GB) — comfortable.
#
# After all tasks finish, run:
#   python scripts/compare_lm_sample_dd.py
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_lm_sample_dd.sbatch
# =============================================================================
#SBATCH --job-name=dd_lm_sample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=4
#SBATCH --time=0-10:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/lm_dd_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/lm_dd_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

echo "===== START LM-SAMPLE DD TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID}  node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate base 2>/dev/null || true

export HF_HOME="/data/user_data/haolingp/hf_cache"
CUDA_DEV=0

# ── Repo & data paths ─────────────────────────────────────────────────────────
REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
cd "${REPO_ROOT}"

SRC="data/enzh/rand100_source.txt"
TGT="data/enzh/rand100_target.txt"
MT_MODEL="facebook/nllb-200-distilled-600M"

# Future LM: Qwen3-4B-Base (local, fast, good English continuation)
# Only generates English futures; NLLB still does the actual EN->ZH translation.
FUTURE_LM="/data/user_data/haolingp/models/Qwen3-4B-Base"
echo "[INFO] Future LM: ${FUTURE_LM}"

# Shared DD hyper-parameters
DD_K=4
DD_STEPS=3
FUTURE_WORDS=15
FUTURE_TEMP=0.9

# ── Per-task config ───────────────────────────────────────────────────────────
declare -a TYPES=( baseline  dd_oracle  dd_lm_sample  dd_lm_sample  dd_oracle )
declare -a WAIT_KS=(  5         5            5              5             5    )
declare -a TAUS=(     0.00      0.05         0.05           0.03          0.03 )

T=${SLURM_ARRAY_TASK_ID}
EXP_TYPE="${TYPES[$T]}"
WAIT_K="${WAIT_KS[$T]}"
TAU="${TAUS[$T]}"

# ── Output directory ──────────────────────────────────────────────────────────
case "${EXP_TYPE}" in
    baseline)     OUT_DIR="${REPO_ROOT}/outputs/lmsdd_baseline_k${WAIT_K}" ;;
    dd_oracle)    OUT_DIR="${REPO_ROOT}/outputs/lmsdd_oracle_tau${TAU}"    ;;
    dd_lm_sample) OUT_DIR="${REPO_ROOT}/outputs/lmsdd_lm_tau${TAU}"       ;;
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
    --model-name "${MT_MODEL}"
    --device   "cuda:${CUDA_DEV}"
    --output   "${OUT_DIR}"
    --beam-size 1
    --uncertainty-threshold 999
)

case "${EXP_TYPE}" in
    baseline)
        EXTRA_ARGS=()
        ;;

    dd_oracle)
        # Oracle truncation futures (existing approach — upper bound)
        EXTRA_ARGS=(
            --dd-gate
            --dd-tau         "${TAU}"
            --dd-futures-k   "${DD_K}"
            --dd-steps       "${DD_STEPS}"
            --dd-future-mode oracle
        )
        ;;

    dd_lm_sample)
        # LM-sampled futures — no oracle required
        # future_lm generates K English continuations; NLLB computes distributions
        EXTRA_ARGS=(
            --dd-gate
            --dd-tau               "${TAU}"
            --dd-futures-k         "${DD_K}"
            --dd-steps             "${DD_STEPS}"
            --dd-future-mode       lm_sample
            --dd-future-lm        "${FUTURE_LM}"
            --dd-future-words      "${FUTURE_WORDS}"
            --dd-future-temperature "${FUTURE_TEMP}"
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
