#!/bin/bash
# =====================================================================
# COMET scoring via SLURM (GPU-accelerated).
#
# Usage:
#   sbatch scripts/run_comet_scoring.sbatch
# =====================================================================

#SBATCH --job-name=comet_score
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=0-04:00:00
#SBATCH --mail-user=haolingp@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/comet_score_%j.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/comet_score_%j.err

source ~/.bashrc
set -euo pipefail

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
mkdir -p "${REPO_ROOT}/outputs/slurm_logs"

# Install COMET if not present
if ! python3 -c "import comet" 2>/dev/null; then
    echo "[COMET] Installing unbabel-comet ..."
    pip install unbabel-comet --quiet
fi

python3 "${REPO_ROOT}/scripts/comet_score_all.py" \
    --search-root "${REPO_ROOT}/outputs" \
    --repo-root "${REPO_ROOT}" \
    --gpus 1
