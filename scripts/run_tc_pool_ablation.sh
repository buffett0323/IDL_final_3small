#!/usr/bin/env bash
# Token Consensus candidate-pool ablation on CoVoST-1997.
# Baseline (top-K=10) already exists at outputs/covost1997/token_consensus_k5
#   -> BLEU 35.845, AL 5.76, COMET 0.8323
# This script adds two variants:
#   (1) top-p=0.9  (nucleus)  -- adaptive pool by cumulative probability
#   (2) min-p=0.1             -- adaptive pool by fraction of peak probability
# Everything else (K=10, wait-k=5, top_logprobs=10) held fixed.
set -e

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"

VLLM_API="http://localhost:8100/v1"
VLLM_MODEL_NAME="qwen30b-instruct"
QWEN4B_PATH="/data/user_data/haolingp/models/Qwen3-4B-Base"
QWEN30B_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

COVOST_SRC="${REPO_ROOT}/data/covost_enzh/subset1997_source_asr.txt"
COVOST_TGT="${REPO_ROOT}/data/covost_enzh/subset1997_target_zh_simul.txt"

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

OUT_ROOT="${REPO_ROOT}/outputs/tc_pool_ablation"
mkdir -p "${OUT_ROOT}"

common_tc=(
    --agent "${REPO_ROOT}/agents/token_consensus_agent.py"
    --source "${COVOST_SRC}" --target "${COVOST_TGT}"
    --wait-k 5
    --future-lm-path "${QWEN4B_PATH}"
    --vllm-api-base "${VLLM_API}"
    --vllm-model-name "${VLLM_MODEL_NAME}"
    --vllm-tokenizer-path "${QWEN30B_PATH}"
    --num-futures 10 --top-logprobs 10 --max-consensus-steps 6
    --base-gpu 0
)

# --- top-p sweep: 0.5, 0.7, 0.9, 0.99 ---
for p in 0.5 0.7 0.9 0.99; do
    run_simul "tc_topp_p${p}" "${OUT_ROOT}/tc_topp_p${p}" \
        "${common_tc[@]}" \
        --pool-mode topp --pool-p "${p}"
done

# --- min-p sweep: 0.05, 0.1, 0.2 ---
for p in 0.05 0.1 0.2; do
    run_simul "tc_minp_p${p}" "${OUT_ROOT}/tc_minp_p${p}" \
        "${common_tc[@]}" \
        --pool-mode minp --pool-p "${p}"
done

echo ""; echo "==================================================================="
echo "[$(date)] TC pool ablation DONE"
echo "Existing baseline (top-K=10): outputs/covost1997/token_consensus_k5 (BLEU 35.845)"
echo "==================================================================="
