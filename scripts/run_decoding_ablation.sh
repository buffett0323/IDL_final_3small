#!/usr/bin/env bash
# Decoding-mode ablation on Qwen direct wait-k=5:
#   (1) top-p sampling (p=0.9)
#   (2) min-p sampling (p=0.1)
# Compared against the existing greedy baseline (BLEU 32.18 on CoVoST-1997,
# BLEU 28.15 on WMT19-1997) produced by outputs/covost1997/qwen_direct_k5 and
# outputs/wmt19_qwen/qwen_direct_k5.
set -e

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"

VLLM_API="http://localhost:8100/v1"
VLLM_MODEL_NAME="qwen30b-instruct"
QWEN4B_PATH="/data/user_data/haolingp/models/Qwen3-4B-Base"

# Datasets
COVOST_SRC="${REPO_ROOT}/data/covost_enzh/subset1997_source_asr.txt"
COVOST_TGT="${REPO_ROOT}/data/covost_enzh/subset1997_target_zh_simul.txt"
WMT_SRC="${REPO_ROOT}/data/enzh/wmt19_source.txt"
WMT_TGT="${REPO_ROOT}/data/enzh/wmt19_target.txt"

# Wait until vLLM is ready
echo "[$(date)] waiting for vLLM on $VLLM_API ..."
for i in $(seq 1 120); do
    if curl -s "$VLLM_API/models" 2>/dev/null | grep -q "$VLLM_MODEL_NAME"; then
        echo "[$(date)] vLLM ready."; break
    fi
    sleep 5
done

run_simul() {
    local tag="$1"; shift
    local out="$1"; shift
    if [[ -f "${out}/scores" ]]; then
        echo "[SKIP ${tag}] ${out}/scores exists"; cat "${out}/scores"; return 0
    fi
    mkdir -p "${out}"
    echo ""; echo "--- [$(date)] RUN ${tag} -> ${out} ---"
    "${SIMULEVAL}" "$@" --output "${out}" 2>&1 | tee "${out}/run.log" | tail -n 2
    if [[ -f "${out}/scores" ]]; then
        echo "[OK ${tag}]"; cat "${out}/scores"
    else
        echo "[FAIL ${tag}]"
    fi
}

common_args=(
    --agent "${REPO_ROOT}/agents/semantic_lcp_agent.py"
    --wait-k 5
    --future-lm-path "${QWEN4B_PATH}"
    --vllm-api-base "${VLLM_API}"
    --vllm-model-name "${VLLM_MODEL_NAME}"
    --num-futures 0
    --future-words 15 --future-temperature 0.9
    --consensus-ratio 0.6
    --base-gpu 0
)

# === CoVoST-1997 ===
OUT_COVOST="${REPO_ROOT}/outputs/decoding_ablation/covost1997"
mkdir -p "${OUT_COVOST}"

run_simul "covost_qwen_topp_p0.9" "${OUT_COVOST}/qwen_topp_p0.9" \
    --source "${COVOST_SRC}" --target "${COVOST_TGT}" \
    "${common_args[@]}" \
    --decode-mode topp --decode-p 0.9

run_simul "covost_qwen_minp_p0.1" "${OUT_COVOST}/qwen_minp_p0.1" \
    --source "${COVOST_SRC}" --target "${COVOST_TGT}" \
    "${common_args[@]}" \
    --decode-mode minp --decode-p 0.1

# === WMT19-1997 ===
OUT_WMT="${REPO_ROOT}/outputs/decoding_ablation/wmt19"
mkdir -p "${OUT_WMT}"

run_simul "wmt_qwen_topp_p0.9" "${OUT_WMT}/qwen_topp_p0.9" \
    --source "${WMT_SRC}" --target "${WMT_TGT}" \
    "${common_args[@]}" \
    --decode-mode topp --decode-p 0.9

run_simul "wmt_qwen_minp_p0.1" "${OUT_WMT}/qwen_minp_p0.1" \
    --source "${WMT_SRC}" --target "${WMT_TGT}" \
    "${common_args[@]}" \
    --decode-mode minp --decode-p 0.1

echo ""; echo "==================================================================="
echo "[$(date)] Decoding-mode ablation DONE"
echo "==================================================================="
