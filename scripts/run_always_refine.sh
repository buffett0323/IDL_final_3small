#!/bin/bash
# New pipeline: Qwen3-4B-Base streaming + Qwen3-30B-Instruct prefix refine at every EOS
#
# Design:
#   - Base model : Qwen3-4B-Base (causal LM, few-shot prompt, GPU0)
#   - Refiner    : Qwen3-30B-A3B-Instruct-2507-FP8 (prefix/assistant mode, GPU1)
#   - Gate       : --always-refine => draft during streaming, Qwen at every EOS
#   - No beam search (beam=1), no LCP, no uncertainty threshold
#
# Usage:
#   bash scripts/run_always_refine.sh [--data rand100|wmt100] [--skip-baseline]

set -e
cd "$(dirname "$0")/.."

BASE_MODEL="/data/user_data/haolingp/models/Qwen3-4B-Base"
QWEN30B="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
K=5

# Default: rand100 (quick)
DATA="rand100"
SKIP_BASELINE=0
for arg in "$@"; do
    case $arg in
        --data) ;;
        rand100|wmt100) DATA=$arg ;;
        --skip-baseline) SKIP_BASELINE=1 ;;
    esac
done

if [ "$DATA" = "wmt100" ]; then
    SOURCE="data/enzh/wmt19_source_100.txt"
    TARGET="data/enzh/wmt19_target_100.txt"
    PREFIX="wmt100"
else
    SOURCE="data/enzh/rand100_source.txt"
    TARGET="data/enzh/rand100_target.txt"
    PREFIX="rand100"
fi

if [ ! -f "$SOURCE" ]; then
    echo "[ERROR] Source not found: $SOURCE"
    echo "Run: python scripts/download_enzh_data.py"
    exit 1
fi

run_exp() {
    local NAME=$1; shift
    local OUT="outputs/${NAME}"
    if [ -d "$OUT" ] && [ -f "${OUT}/scores" ]; then
        echo "[SKIP] $NAME already done"
        cat "${OUT}/scores"
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
        --base-gpu 0 \
        --trace-refinement \
        "$@" \
        --output "$OUT"
    echo "[DONE] $NAME"
    cat "${OUT}/scores"
}

# ---- 1. NLLB baseline (reference point, skip if already done) ----
if [ $SKIP_BASELINE -eq 0 ] && [ ! -f "outputs/${PREFIX}_baseline_k5/scores" ]; then
    run_exp "${PREFIX}_baseline_k5" \
        --wait-k $K \
        --uncertainty-threshold 999
fi

# ---- 2. Qwen4B-Base streaming only (no refiner) ----
run_exp "${PREFIX}_qwen4b_base_only" \
    --model-name "$BASE_MODEL" \
    --causal-lm \
    --wait-k $K \
    --uncertainty-threshold 999

# ---- 3. Qwen4B-Base + Qwen30B-Instruct prefix refine (always at EOS) ----
run_exp "${PREFIX}_qwen4b_qwen30b_always" \
    --model-name "$BASE_MODEL" \
    --causal-lm \
    --wait-k $K \
    --always-refine \
    --qwen-model-path "$QWEN30B" \
    --qwen-gpu 1 \
    --qwen-mode prefix

# ---- 4. NLLB + Qwen30B-Instruct prefix refine (always at EOS) — ablation ----
run_exp "${PREFIX}_nllb_qwen30b_always" \
    --wait-k $K \
    --always-refine \
    --qwen-model-path "$QWEN30B" \
    --qwen-gpu 1 \
    --qwen-mode prefix

# ---- Summary ----
echo ""
echo "============================================"
echo "Summary (BLEU / AL):"
echo "============================================"
for name in \
    "${PREFIX}_baseline_k5" \
    "${PREFIX}_qwen4b_base_only" \
    "${PREFIX}_qwen4b_qwen30b_always" \
    "${PREFIX}_nllb_qwen30b_always" \
    "${PREFIX}_sttr_k5_tau3.0" \
    "${PREFIX}_sttr_k5_tau3.0_qwen"; do
    dir="outputs/${name}"
    if [ -f "${dir}/scores" ]; then
        python3 - <<PYEOF
import re
txt = open('${dir}/scores').read()
nums = re.findall(r'[\d]+\.[\d]+', txt)
bleu = nums[0] if nums else 'N/A'
al   = nums[2] if len(nums) > 2 else 'N/A'
print(f"  {'${name}':<48}  BLEU={bleu:<8}  AL={al}")
PYEOF
    fi
done

# ---- COMET evaluation ----
echo ""
echo "============================================"
echo "COMET evaluation ..."
echo "============================================"
COMET_DIRS=()
for name in \
    "${PREFIX}_baseline_k5" \
    "${PREFIX}_qwen4b_base_only" \
    "${PREFIX}_qwen4b_qwen30b_always" \
    "${PREFIX}_nllb_qwen30b_always" \
    "${PREFIX}_sttr_k5_tau3.0_qwen"; do
    dir="outputs/${name}"
    [ -f "${dir}/scores" ] && COMET_DIRS+=("$dir")
done

if [ ${#COMET_DIRS[@]} -gt 0 ]; then
    python3 scripts/eval_comet.py "${COMET_DIRS[@]}" \
        --output "outputs/${PREFIX}_always_refine_comet.json"
fi
