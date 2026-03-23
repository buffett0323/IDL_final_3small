#!/bin/bash
# Experiments with Qwen3-4B-Base as the base translation model
# Three settings:
#   1. qwen4b_baseline     – Qwen3-4B-Base only, no Qwen30B refiner
#   2. qwen4b_qwen_eos     – Qwen3-4B-Base + Qwen30B prefix-continuation at every EOS
#   3. qwen4b_qwen_rerank  – Qwen3-4B-Base + Qwen30B log-prob reranker at every EOS
#
# Usage:
#   bash scripts/run_qwen4b.sh

set -e

SOURCE="data/enzh/rand100_source.txt"
TARGET="data/enzh/rand100_target.txt"
BASE_MODEL="/data/user_data/haolingp/models/Qwen3-4B-Base"
QWEN30B="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
K=5
MODE="tail3"
TAU=3.0

run_exp() {
    local NAME=$1; shift
    local OUT="outputs/${NAME}"
    if [ -d "$OUT" ] && [ -f "${OUT}/scores" ]; then
        echo "[SKIP] $NAME already done"
        return
    fi
    echo ""
    echo "============================================"
    echo "Running: $NAME"
    echo "============================================"
    rm -rf "$OUT"
    simuleval \
        --agent agents/sttr_enzh_agent.py \
        --source "$SOURCE" --target "$TARGET" \
        --source-lang eng_Latn --target-lang zho_Hans \
        --base-gpu 0 --trace-refinement \
        "$@" \
        --output "$OUT"
    echo "[DONE] $NAME"
    cat "${OUT}/scores"
}

# 1. Baseline: Qwen3-4B-Base streaming, no Qwen30B refinement
run_exp "qwen4b_baseline" \
    --model-name "$BASE_MODEL" \
    --causal-lm \
    --wait-k $K \
    --uncertainty-threshold 999 \
    --uncertainty-mode $MODE

# 2. Qwen3-4B-Base + Qwen30B prefix-continuation at every EOS
run_exp "qwen4b_qwen_eos" \
    --model-name "$BASE_MODEL" \
    --causal-lm \
    --wait-k $K \
    --uncertainty-threshold $TAU \
    --uncertainty-mode $MODE \
    --num-candidates 4 --max-extra-reads 3 \
    --qwen-model-path "$QWEN30B" \
    --qwen-gpu 1 \
    --qwen-mode prefix

# 3. Qwen3-4B-Base + Qwen30B log-prob reranker at every EOS
run_exp "qwen4b_qwen_rerank" \
    --model-name "$BASE_MODEL" \
    --causal-lm \
    --wait-k $K \
    --uncertainty-threshold $TAU \
    --uncertainty-mode $MODE \
    --num-candidates 4 --max-extra-reads 3 \
    --qwen-model-path "$QWEN30B" \
    --qwen-gpu 1 \
    --qwen-mode logprob_rerank

echo ""
echo "============================================"
echo "All Qwen4B experiments done! Summary:"
echo "============================================"
for dir in outputs/qwen4b_*/ outputs/rand100_baseline_k5/; do
    if [ -f "${dir}scores" ]; then
        python3 -c "
import re, os
name = '$(basename $dir 2>/dev/null || echo unknown)'
try:
    txt = open('${dir}scores').read()
    nums = re.findall(r'[\d]+\.[\d]+', txt)
    bleu = nums[0] if nums else 'N/A'
    al   = nums[2] if len(nums)>2 else 'N/A'
    print(f'  {name:<45}  BLEU={bleu:<8}  AL={al}')
except: pass
"
    fi
done

echo ""
echo "Running COMET evaluation..."
python3 scripts/eval_comet.py \
    outputs/rand100_baseline_k5 \
    outputs/rand100_sttr_k5_tau3.0 \
    outputs/rand100_sttr_k5_tau3.0_qwen \
    outputs/qwen4b_baseline \
    outputs/qwen4b_qwen_eos \
    outputs/qwen4b_qwen_rerank \
    --output outputs/qwen4b_comet_results.json
