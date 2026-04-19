#!/usr/bin/env bash
# =============================================================================
# run_wmt500_nllb_baseline.sbatch
#
# NLLB wait-k=5 **pure baseline** on **wmt500** (500 sentences) — same slice as
# Qwen / Semantic LCP in `run_full_qwen.sbatch`, for fair same-N comparisons.
#
# Output: outputs/full/nllb_baseline_k5_wmt500/
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_wmt500_nllb_baseline.sbatch
# =============================================================================
#SBATCH --job-name=wmt500_nllb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=0-06:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/wmt500_nllb_%j.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/wmt500_nllb_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haolingp@andrew.cmu.edu

set -o pipefail

echo "===== WMT500 NLLB BASELINE k=5 ====="
echo "job_id=${SLURM_JOB_ID}  node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true

source ~/.bashrc
conda activate base 2>/dev/null || true

export HF_HOME="/data/user_data/haolingp/hf_cache"
CUDA_DEV=0

CONDA_BASE="${CONDA_BASE:-/home/haolingp/miniconda3}"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
cd "${REPO_ROOT}"

SRC="${REPO_ROOT}/data/enzh/wmt500_source.txt"
TGT="${REPO_ROOT}/data/enzh/wmt500_target.txt"
OUT_DIR="${REPO_ROOT}/outputs/full/nllb_baseline_k5_wmt500"
NLLB_MODEL="facebook/nllb-200-distilled-600M"
WAIT_K=5

mkdir -p "${OUT_DIR}" "${REPO_ROOT}/outputs/slurm_logs"

echo "Data   : ${SRC} / ${TGT}"
echo "Output : ${OUT_DIR}"

if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] Already complete."
    cat "${OUT_DIR}/scores"
    exit 0
fi

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
    --uncertainty-threshold 999
)

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" "${SIMULEVAL}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo "===== DONE ====="
[[ -f "${OUT_DIR}/scores" ]] && cat "${OUT_DIR}/scores"
