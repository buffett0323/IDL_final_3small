#!/usr/bin/env bash
# =============================================================================
# run_full_nllb_rerun_prefixanchored.sbatch
# Re-run the NLLB supporting line after switching sttr_enzh_agent.py to
# prefix-anchored commit semantics.
#
# Outputs go to:
#   outputs/rerun_20260330/nllb_support/
#
# Array layout:
#   0  NLLB baseline        wait-k=3
#   1  NLLB baseline        wait-k=5
#   2  NLLB baseline        wait-k=7
#   3  NLLB baseline        wait-k=9
#   4  NLLB DD full gate    wait-k=5  tau=0.05
#   5  NLLB DD veto         wait-k=5  tau=0.03
#   6  NLLB LM-sampled DD   wait-k=5  tau=0.03
# =============================================================================
#SBATCH --job-name=nllb_rerun_pa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-6
#SBATCH --time=0-16:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/nllb_rerun_pa_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/nllb_rerun_pa_%A_%a.err

set -e

source ~/.bashrc
conda activate base 2>/dev/null || true

export HF_HOME="/data/user_data/haolingp/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
MODEL_ROOT="/data/user_data/haolingp/models"
OUT_ROOT="${REPO_ROOT}/outputs/rerun_20260330/nllb_support"
cd "${REPO_ROOT}"

SRC="data/enzh/wmt19_source.txt"
TGT="data/enzh/wmt19_target.txt"
NLLB_MODEL="facebook/nllb-200-distilled-600M"
QWEN4B_PATH="${MODEL_ROOT}/Qwen3-4B-Base"
CUDA_DEV=0
DD_K=4
DD_STEPS=3

declare -a TYPES=(   baseline  baseline  baseline  baseline  dd_full   dd_veto   lm_dd  )
declare -a WAIT_KS=( 3         5         7         9         5         5         5      )
declare -a TAUS=(    0.00      0.00      0.00      0.00      0.05      0.03      0.03   )

T=${SLURM_ARRAY_TASK_ID}
EXP_TYPE="${TYPES[$T]}"
WAIT_K="${WAIT_KS[$T]}"
TAU="${TAUS[$T]}"

mkdir -p "${OUT_ROOT}" "${REPO_ROOT}/outputs/slurm_logs"

case "${EXP_TYPE}" in
    baseline) OUT_DIR="${OUT_ROOT}/nllb_baseline_k${WAIT_K}" ;;
    dd_full)  OUT_DIR="${OUT_ROOT}/nllb_dd_full_tau${TAU}"   ;;
    dd_veto)  OUT_DIR="${OUT_ROOT}/nllb_dd_veto_tau${TAU}"   ;;
    lm_dd)    OUT_DIR="${OUT_ROOT}/nllb_lmsdd_tau${TAU}"     ;;
esac

mkdir -p "${OUT_DIR}"

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
            --dd-tau "${TAU}"
            --dd-futures-k "${DD_K}"
            --dd-steps "${DD_STEPS}"
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
            --dd-tau "${TAU}"
            --dd-futures-k "${DD_K}"
            --dd-steps "${DD_STEPS}"
            --dd-future-lm "${QWEN4B_PATH}"
            --dd-future-words 15
            --dd-future-temperature 0.9
        )
        ;;
    lm_dd)
        EXTRA_ARGS=(
            --uncertainty-threshold 999
            --dd-gate
            --dd-tau "${TAU}"
            --dd-futures-k "${DD_K}"
            --dd-steps "${DD_STEPS}"
            --dd-future-lm "${QWEN4B_PATH}"
            --dd-future-words 15
            --dd-future-temperature 0.9
        )
        ;;
esac

echo "[RUN] ${EXP_TYPE} wait-k=${WAIT_K} tau=${TAU}"
CUDA_VISIBLE_DEVICES="${CUDA_DEV}" "${SIMULEVAL}" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"
