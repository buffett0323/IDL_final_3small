#!/usr/bin/env bash
# Master sequential runner for 6 methods on CoVoST-1997 (match WMT19 scale).
set -e

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"

VLLM_PORT="${VLLM_PORT:-$(cat ${REPO_ROOT}/outputs/_master_runs/vllm_covost500.port)}"
VLLM_API="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="qwen30b-instruct"

QWEN4B_PATH="/data/user_data/haolingp/models/Qwen3-4B-Base"
QWEN30B_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
NLLB_MODEL="facebook/nllb-200-distilled-600M"

AGENT_GPU=1

export HF_HOME="/data/user_data/haolingp/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${AGENT_GPU}"

MASTER_LOG="${REPO_ROOT}/outputs/_master_runs/covost1997_master.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========================================================================="
echo "[$(date)] CoVoST-1997 MASTER RUNNER START (match WMT19 scale)"
echo "  VLLM_API=${VLLM_API}"
echo "========================================================================="

SRC="${REPO_ROOT}/data/covost_enzh/subset1997_source_asr.txt"
TGT="${REPO_ROOT}/data/covost_enzh/subset1997_target_zh_simul.txt"
for f in "${SRC}" "${TGT}"; do
    if [[ ! -s "${f}" ]]; then
        echo "ERROR: input missing: ${f}"
        exit 1
    fi
done
echo "SRC=${SRC} ($(wc -l < ${SRC}) lines)"
echo "TGT=${TGT} ($(wc -l < ${TGT}) lines)"

if ! curl -s "${VLLM_API}/models" | grep -q "${VLLM_MODEL_NAME}"; then
    echo "ERROR: vLLM not responding at ${VLLM_API}"
    exit 1
fi

OUT_ROOT="${REPO_ROOT}/outputs/covost1997"
mkdir -p "${OUT_ROOT}"

run_simul() {
    local tag="$1"; shift
    local out="$1"; shift
    if [[ -f "${out}/scores" ]]; then
        echo "[SKIP ${tag}] ${out}/scores exists"
        cat "${out}/scores"
        return 0
    fi
    mkdir -p "${out}"
    echo ""
    echo "--- [$(date)] RUN ${tag} -> ${out} ---"
    "${SIMULEVAL}" "$@" --output "${out}" 2>&1 | tee "${out}/run.log" | tail -n 2
    if [[ -f "${out}/scores" ]]; then
        echo "[OK ${tag}]"
        cat "${out}/scores"
    else
        echo "[FAIL ${tag}]"
    fi
}

# (1) NLLB greedy
run_simul "nllb_baseline_k5" "${OUT_ROOT}/nllb_baseline_k5" \
    --agent "${REPO_ROOT}/agents/sttr_enzh_agent.py" \
    --source "${SRC}" --target "${TGT}" \
    --source-lang eng_Latn --target-lang zho_Hans \
    --wait-k 5 \
    --model-name "${NLLB_MODEL}" \
    --device "cuda:0" \
    --beam-size 1 \
    --uncertainty-threshold 999

# (2) NLLB DD veto
run_simul "nllb_dd_veto" "${OUT_ROOT}/nllb_dd_veto_tau0.03" \
    --agent "${REPO_ROOT}/agents/sttr_enzh_agent.py" \
    --source "${SRC}" --target "${TGT}" \
    --source-lang eng_Latn --target-lang zho_Hans \
    --wait-k 5 \
    --model-name "${NLLB_MODEL}" \
    --device "cuda:0" \
    --beam-size 1 \
    --uncertainty-threshold 3.0 --uncertainty-mode tail3 \
    --dd-veto --dd-tau 0.03 --dd-futures-k 4 --dd-steps 3 \
    --dd-future-lm "${QWEN4B_PATH}" \
    --dd-future-words 15 --dd-future-temperature 0.9

# (3) Qwen direct
run_simul "qwen_direct" "${OUT_ROOT}/qwen_direct_k5" \
    --agent "${REPO_ROOT}/agents/semantic_lcp_agent.py" \
    --source "${SRC}" --target "${TGT}" \
    --wait-k 5 \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --num-futures 0 \
    --future-words 15 --future-temperature 0.9 \
    --consensus-ratio 0.6 \
    --base-gpu 0

# (4) Qwen DD+JS
run_simul "qwen_dd" "${OUT_ROOT}/qwen_dd_k5_f4_tau0.15" \
    --agent "${REPO_ROOT}/agents/semantic_lcp_agent.py" \
    --source "${SRC}" --target "${TGT}" \
    --wait-k 5 \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --num-futures 4 \
    --future-words 15 --future-temperature 0.9 \
    --consensus-ratio 0.6 \
    --gate-js --gate-tau 0.15 --gate-steps 3 \
    --base-gpu 0

# (5) SemLCP K=4
run_simul "semlcp" "${OUT_ROOT}/semlcp_k5_f4" \
    --agent "${REPO_ROOT}/agents/semantic_lcp_agent.py" \
    --source "${SRC}" --target "${TGT}" \
    --wait-k 5 \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --num-futures 4 \
    --future-words 15 --future-temperature 0.9 \
    --consensus-ratio 0.6 \
    --base-gpu 0

# (6) Token Consensus K=10 top=10
run_simul "token_consensus" "${OUT_ROOT}/token_consensus_k5" \
    --agent "${REPO_ROOT}/agents/token_consensus_agent.py" \
    --source "${SRC}" --target "${TGT}" \
    --wait-k 5 \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --vllm-tokenizer-path "${QWEN30B_PATH}" \
    --num-futures 10 --top-logprobs 10 --max-consensus-steps 6 \
    --base-gpu 0

echo ""
echo "========================================================================="
echo "[$(date)] CoVoST-1997 MASTER RUNNER DONE"
echo "========================================================================="
