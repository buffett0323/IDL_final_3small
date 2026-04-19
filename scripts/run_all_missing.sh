#!/usr/bin/env bash
# Master sequential runner for all missing experiments.
# Assumes vLLM 30B is already up on GPU 0 at port ${VLLM_PORT}.
# Agent jobs run on GPU 1.
#
# Usage:  bash scripts/run_all_missing.sh
set -e

REPO_ROOT="/data/user_data/haolingp/IDL_final_3small"
CONDA_BASE="/home/haolingp/miniconda3"
SIMULEVAL="${CONDA_BASE}/bin/simuleval"
PY="${CONDA_BASE}/bin/python"

VLLM_PORT="${VLLM_PORT:-$(cat ${REPO_ROOT}/outputs/_master_runs/vllm_30b_master.port)}"
VLLM_API="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="qwen30b-instruct"

QWEN4B_PATH="/data/user_data/haolingp/models/Qwen3-4B-Base"
QWEN30B_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
NLLB_MODEL="facebook/nllb-200-distilled-600M"

# Agent GPU — vLLM is on GPU 0, so agent runs on GPU 1
AGENT_GPU=1

export HF_HOME="/data/user_data/haolingp/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${AGENT_GPU}"

MASTER_LOG="${REPO_ROOT}/outputs/_master_runs/master.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========================================================================="
echo "[$(date)] MASTER RUNNER START"
echo "  VLLM_API=${VLLM_API}"
echo "  AGENT_GPU=${AGENT_GPU} (physical)"
echo "========================================================================="

# Sanity-check vLLM is responsive
if ! curl -s "${VLLM_API}/models" | grep -q "${VLLM_MODEL_NAME}"; then
  echo "ERROR: vLLM not responding at ${VLLM_API}"
  exit 1
fi

# rand100 and covost data paths
RAND_SRC="${REPO_ROOT}/data/enzh/rand100_source.txt"
RAND_TGT="${REPO_ROOT}/data/enzh/rand100_target.txt"
COVOST_SRC="${REPO_ROOT}/data/covost_enzh/subset100_source_asr.txt"
COVOST_TGT="${REPO_ROOT}/data/covost_enzh/subset100_target_zh_simul.txt"

# ---------------------------------------------------------------------------
# Helper: run a SimulEval job unless already complete
# ---------------------------------------------------------------------------
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

# ===========================================================================
# PHASE A: TC top-k ablation (rand100)
# ===========================================================================
TC_AGENT="${REPO_ROOT}/agents/token_consensus_agent.py"

for TOP_K in 5 20; do
    OUT="${REPO_ROOT}/outputs/tc_ablation/tc_k5_K10_top${TOP_K}"
    # Clean stale partial state from prior incomplete run (only for top5)
    if [[ -f "${OUT}/token_consensus_trace.jsonl" && ! -f "${OUT}/scores" ]]; then
        echo "[CLEAN] Removing stale partial dir ${OUT}"
        rm -rf "${OUT}"
    fi
    run_simul "tc_k5_K10_top${TOP_K}" "${OUT}" \
        --agent "${TC_AGENT}" \
        --source "${RAND_SRC}" --target "${RAND_TGT}" \
        --wait-k 5 \
        --future-lm-path "${QWEN4B_PATH}" \
        --vllm-api-base "${VLLM_API}" \
        --vllm-model-name "${VLLM_MODEL_NAME}" \
        --vllm-tokenizer-path "${QWEN30B_PATH}" \
        --num-futures 10 \
        --top-logprobs "${TOP_K}" \
        --max-consensus-steps 6 \
        --base-gpu 0
done

# ===========================================================================
# PHASE B: CoVoST cascaded — NLLB-based DD, Qwen SemLCP, Qwen TC
# ===========================================================================
COVOST_OUT_ROOT="${REPO_ROOT}/outputs/covost_cascaded"

# (1) NLLB DD veto on CoVoST ASR (doesn't need vLLM; uses Qwen4B future LM + NLLB locally)
OUT="${COVOST_OUT_ROOT}/nllb_dd_veto_tau0.03_asr_subset100"
run_simul "covost_nllb_dd_veto" "${OUT}" \
    --agent "${REPO_ROOT}/agents/sttr_enzh_agent.py" \
    --source "${COVOST_SRC}" --target "${COVOST_TGT}" \
    --source-lang eng_Latn --target-lang zho_Hans \
    --wait-k 5 \
    --model-name "${NLLB_MODEL}" \
    --device "cuda:0" \
    --beam-size 1 \
    --uncertainty-threshold 3.0 \
    --uncertainty-mode tail3 \
    --dd-veto \
    --dd-tau 0.03 \
    --dd-futures-k 4 \
    --dd-steps 3 \
    --dd-future-lm "${QWEN4B_PATH}" \
    --dd-future-words 15 \
    --dd-future-temperature 0.9

# (2) Qwen30B direct baseline on CoVoST ASR (for fair comparison with Qwen methods)
OUT="${COVOST_OUT_ROOT}/qwen_direct_k5_asr_subset100"
run_simul "covost_qwen_direct" "${OUT}" \
    --agent "${REPO_ROOT}/agents/semantic_lcp_agent.py" \
    --source "${COVOST_SRC}" --target "${COVOST_TGT}" \
    --wait-k 5 \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --num-futures 0 \
    --future-words 15 \
    --future-temperature 0.9 \
    --consensus-ratio 0.6 \
    --base-gpu 0

# (3) Qwen DD+JS gate on CoVoST ASR
OUT="${COVOST_OUT_ROOT}/qwen_dd_k5_f4_tau0.15_asr_subset100"
run_simul "covost_qwen_dd" "${OUT}" \
    --agent "${REPO_ROOT}/agents/semantic_lcp_agent.py" \
    --source "${COVOST_SRC}" --target "${COVOST_TGT}" \
    --wait-k 5 \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --num-futures 4 \
    --future-words 15 \
    --future-temperature 0.9 \
    --consensus-ratio 0.6 \
    --gate-js --gate-tau 0.15 --gate-steps 3 \
    --base-gpu 0

# (4) SemLCP on CoVoST ASR
OUT="${COVOST_OUT_ROOT}/semlcp_k5_f4_asr_subset100"
run_simul "covost_semlcp" "${OUT}" \
    --agent "${REPO_ROOT}/agents/semantic_lcp_agent.py" \
    --source "${COVOST_SRC}" --target "${COVOST_TGT}" \
    --wait-k 5 \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --num-futures 4 \
    --future-words 15 \
    --future-temperature 0.9 \
    --consensus-ratio 0.6 \
    --base-gpu 0

# (5) Token Consensus on CoVoST ASR
OUT="${COVOST_OUT_ROOT}/token_consensus_k5_asr_subset100"
run_simul "covost_tc" "${OUT}" \
    --agent "${REPO_ROOT}/agents/token_consensus_agent.py" \
    --source "${COVOST_SRC}" --target "${COVOST_TGT}" \
    --wait-k 5 \
    --future-lm-path "${QWEN4B_PATH}" \
    --vllm-api-base "${VLLM_API}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --vllm-tokenizer-path "${QWEN30B_PATH}" \
    --num-futures 10 \
    --top-logprobs 10 \
    --max-consensus-steps 6 \
    --base-gpu 0

echo ""
echo "========================================================================="
echo "[$(date)] MASTER RUNNER DONE"
echo "========================================================================="
