#!/usr/bin/env bash
# =============================================================================
# run_entropy_only.sbatch
# Control experiment: entropy gate only (no DD), wait-k=5
#
# Purpose: isolate the DD contribution from DD veto results.
#   DD veto uses: wait-k=5 + entropy gate (tau=3.0) + DD veto
#   This run uses: wait-k=5 + entropy gate (tau=3.0) only (no DD at all)
#
# Comparing this vs DD veto gives the NET contribution of DD alone.
#
# Output: outputs/cmp_entropy_only/
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_entropy_only.sbatch
# =============================================================================
#SBATCH --job-name=entropy_only
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=0-04:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/entropy_only_%j.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/entropy_only_%j.err

set -e

echo "===== START entropy_only ====="
echo "job_id=${SLURM_JOB_ID}  node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

source ~/.bashrc
conda activate base 2>/dev/null || true
export HF_HOME="/data/user_data/haolingp/hf_cache"

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
cd "${REPO_ROOT}"

OUT_DIR="${REPO_ROOT}/outputs/cmp_entropy_only"
mkdir -p "${REPO_ROOT}/outputs/slurm_logs"

if [[ -f "${OUT_DIR}/scores" ]]; then
    echo "[SKIP] Already complete. Delete outputs/cmp_entropy_only to re-run."
    cat "${OUT_DIR}/scores"
    exit 0
fi

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "Running: wait-k=5, entropy gate tau=3.0, NO DD gate/veto"

CUDA_VISIBLE_DEVICES=0 simuleval \
    --agent    "${REPO_ROOT}/agents/sttr_enzh_agent.py" \
    --source   "data/enzh/rand100_source.txt" \
    --target   "data/enzh/rand100_target.txt" \
    --source-lang eng_Latn \
    --target-lang zho_Hans \
    --wait-k   5 \
    --model-name "facebook/nllb-200-distilled-600M" \
    --device   "cuda:0" \
    --output   "${OUT_DIR}" \
    --beam-size 1 \
    --uncertainty-threshold 3.0 \
    --uncertainty-mode tail3 \
    --max-extra-reads 3 \
    2>&1 | tee "${OUT_DIR}/run.log"

echo ""
echo "[DONE] entropy_only"
[[ -f "${OUT_DIR}/scores" ]] && echo "Scores:" && cat "${OUT_DIR}/scores"
echo "===== DONE ====="
