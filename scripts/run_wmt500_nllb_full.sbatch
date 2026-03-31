#!/usr/bin/env bash
# =============================================================================
# run_wmt500_nllb_full.sbatch
# NLLB experiments on wmt500 (500 sentences) — for fair two-group comparison
#
# Array layout:
#   0  NLLB baseline  wait-k=3   beam=1   (Baseline 1 low latency)
#   1  NLLB baseline  wait-k=7   beam=1   (Baseline 1 high latency)
#   2  NLLB baseline  wait-k=5   beam=8   (Baseline 2)
#   3  NLLB DD full   wait-k=5   tau=0.05 (Method 1 variant)
#   4  NLLB DD veto   wait-k=5   tau=0.03 (Method 1 best)
#   5  NLLB literal LCP wait-k=5 K=4      (Method 2 small-model track)
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_wmt500_nllb_full.sbatch
# =============================================================================
#SBATCH --job-name=wmt500_nllb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=60G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-5
#SBATCH --time=0-08:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/wmt500_nllb_%A_%a.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/wmt500_nllb_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -e

echo "===== WMT500 NLLB TASK ${SLURM_ARRAY_TASK_ID} ====="
echo "job_id=${SLURM_JOB_ID}  node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
export HF_HOME="/data/user_data/haolingp/hf_cache"
CUDA_DEV=0

CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
MODEL_ROOT="/data/user_data/haolingp/models"
cd "${REPO_ROOT}"

# wmt500: 500-sentence subset (same as Qwen experiments → directly comparable)
SRC="data/enzh/wmt500_source.txt"
TGT="data/enzh/wmt500_target.txt"
NLLB_MODEL="facebook/nllb-200-distilled-600M"
QWEN4B_PATH="${MODEL_ROOT}/Qwen3-4B-Base"

DD_K=4
DD_STEPS=3

# ── Per-task config ───────────────────────────────────────────────────────────
declare -a TYPES=(    baseline  baseline  baseline_b8  dd_full  dd_veto  literal_lcp )
declare -a WAIT_KS=(  3         7         5            5        5        5           )
declare -a BEAM_SZS=( 1         1         8            1        1        1           )
declare -a TAUS=(     0.00      0.00      0.00         0.05     0.03     0.00        )

T=${SLURM_ARRAY_TASK_ID}
EXP_TYPE="${TYPES[$T]}"
WAIT_K="${WAIT_KS[$T]}"
BEAM_SZ="${BEAM_SZS[$T]}"
TAU="${TAUS[$T]}"

# ── Output directory ──────────────────────────────────────────────────────────
mkdir -p "${REPO_ROOT}/outputs/full"
mkdir -p "${REPO_ROOT}/outputs/slurm_logs"

case "${EXP_TYPE}" in
    baseline)     OUT_DIR="${REPO_ROOT}/outputs/full/nllb_baseline_k${WAIT_K}_wmt500" ;;
    baseline_b8)  OUT_DIR="${REPO_ROOT}/outputs/full/nllb_baseline_k5_beam8_wmt500"   ;;
    dd_full)      OUT_DIR="${REPO_ROOT}/outputs/full/nllb_dd_full_tau${TAU}_wmt500"   ;;
    dd_veto)      OUT_DIR="${REPO_ROOT}/outputs/full/nllb_dd_veto_tau${TAU}_wmt500"   ;;
    literal_lcp)  OUT_DIR="${REPO_ROOT}/outputs/full/nllb_lcp_k${WAIT_K}_f${DD_K}_wmt500" ;;
esac

mkdir -p "${OUT_DIR}"

echo "──────────────────────────────────────────────────────"
echo "Experiment : ${EXP_TYPE}  wait-k=${WAIT_K}  beam=${BEAM_SZ}  tau=${TAU}"
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
    --beam-size   "${BEAM_SZ}"
)

case "${EXP_TYPE}" in
    baseline|baseline_b8)
        EXTRA_ARGS=(--uncertainty-threshold 999)
        ;;
    dd_full)
        EXTRA_ARGS=(
            --uncertainty-threshold 999
            --dd-gate
            --dd-tau            "${TAU}"
            --dd-futures-k      "${DD_K}"
            --dd-steps          "${DD_STEPS}"
            --dd-future-lm      "${QWEN4B_PATH}"
            --dd-future-words   15
            --dd-future-temperature 0.9
        )
        ;;
    dd_veto)
        EXTRA_ARGS=(
            --uncertainty-threshold 3.0
            --uncertainty-mode  tail3
            --dd-veto
            --dd-tau            "${TAU}"
            --dd-futures-k      "${DD_K}"
            --dd-steps          "${DD_STEPS}"
            --dd-future-lm      "${QWEN4B_PATH}"
            --dd-future-words   15
            --dd-future-temperature 0.9
        )
        ;;
    literal_lcp)
        EXTRA_ARGS=(
            --uncertainty-threshold 999
            --always-refine
            --num-candidates "${DD_K}"
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
echo "[DONE] ${EXP_TYPE}  wait-k=${WAIT_K}  beam=${BEAM_SZ}  tau=${TAU}"
if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "=== SCORES ==="
    cat "${OUT_DIR}/scores"
fi
echo "===== DONE TASK ${SLURM_ARRAY_TASK_ID} ====="
