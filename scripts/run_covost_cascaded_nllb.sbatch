#!/usr/bin/env bash
# =============================================================================
# Cascaded ST on CoVoST EN→ZH (HF mirror): Speech → Whisper ASR → SimulEval (NLLB).
#
# Step 1: prepare_covost_asr_texts.py  (GPU, downloads HF data + runs Whisper)
# Step 2: simuleval sttr_enzh_agent    (GPU, wait-k baseline)
#
# Submit:
#   cd /data/user_data/haolingp/IDL_final_3small
#   sbatch scripts/run_covost_cascaded_nllb.sbatch
#
# Override (full test split — long):
#   sbatch --export=HF_SPLIT=test,PREFIX=fulltest,WHISPER=openai/whisper-small \
#     scripts/run_covost_cascaded_nllb.sbatch
# =============================================================================
#SBATCH --job-name=covost_cascaded
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=0-12:00:00
#SBATCH -o /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/covost_cascaded_%j.out
#SBATCH -e /data/user_data/haolingp/IDL_final_3small/outputs/slurm_logs/covost_cascaded_%j.err

# NOTE: Do not use `set -u` here. Sourcing ~/.bashrc under `-u` can abort the whole
# batch job immediately (unset vars in bashrc) with exit 1 and ~0s runtime — no SimulEval logs.
set -eo pipefail

HF_SPLIT="${HF_SPLIT:-test[:100]}"
PREFIX="${PREFIX:-subset100}"
WHISPER="${WHISPER:-openai/whisper-small}"
WAIT_K="${WAIT_K:-5}"

echo "===== CoVoST cascaded ST  job_id=${SLURM_JOB_ID}  $(date) ====="
echo "HF_SPLIT=${HF_SPLIT}  PREFIX=${PREFIX}  WHISPER=${WHISPER}  WAIT_K=${WAIT_K}"
echo "node=$(hostname)  pwd=$(pwd)"

# Conda: prefer explicit init (same idea as run_full_nllb.sbatch, but avoid relying on bashrc).
CONDA_BASE="/home/haolingp/miniconda3"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate base 2>/dev/null || true
else
  source ~/.bashrc 2>/dev/null || true
  conda activate base 2>/dev/null || true
fi

echo "[env] python=$("${CONDA_BASE}/bin/python" -c 'import sys; print(sys.executable)')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

export HF_HOME="${HF_HOME:-/data/user_data/haolingp/hf_cache}"

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"
PY="${CONDA_BASE}/bin/python"
cd "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/outputs/slurm_logs"
mkdir -p "${REPO_ROOT}/data/covost_enzh"

DATA_DIR="${REPO_ROOT}/data/covost_enzh"
SRC="${DATA_DIR}/${PREFIX}_source_asr.txt"
TGT="${DATA_DIR}/${PREFIX}_target_zh.txt"

if [[ -s "${SRC}" && -s "${TGT}" ]]; then
  echo "[SKIP ASR] Using existing ${SRC} / ${TGT}"
else
  echo "[RUN ASR] prepare_covost_asr_texts.py ..."
  CUDA_VISIBLE_DEVICES=0 "${PY}" "${REPO_ROOT}/scripts/prepare_covost_asr_texts.py" \
    --hf-split "${HF_SPLIT}" \
    --prefix "${PREFIX}" \
    --whisper-model "${WHISPER}" \
    --device cuda:0 \
    --dtype float16
fi

OUT_DIR="${REPO_ROOT}/outputs/covost_cascaded/nllb_baseline_k${WAIT_K}_asr_${PREFIX}"
mkdir -p "${OUT_DIR}"

if [[ -f "${OUT_DIR}/scores" ]]; then
  echo "[SKIP SimulEval] ${OUT_DIR}/scores exists"
  cat "${OUT_DIR}/scores"
  exit 0
fi

NLLB_MODEL="facebook/nllb-200-distilled-600M"

echo "[RUN] simuleval to ${OUT_DIR}"
CUDA_VISIBLE_DEVICES=0 "${SIMULEVAL}" \
  --agent       "${REPO_ROOT}/agents/sttr_enzh_agent.py" \
  --source      "${SRC}" \
  --target      "${TGT}" \
  --source-lang eng_Latn \
  --target-lang zho_Hans \
  --wait-k      "${WAIT_K}" \
  --model-name  "${NLLB_MODEL}" \
  --device      "cuda:0" \
  --output      "${OUT_DIR}" \
  --beam-size   1 \
  --uncertainty-threshold 999 \
  2>&1 | tee "${OUT_DIR}/run.log"

echo "===== DONE $(date) ====="
if [[ -f "${OUT_DIR}/scores" ]]; then
  cat "${OUT_DIR}/scores"
fi
