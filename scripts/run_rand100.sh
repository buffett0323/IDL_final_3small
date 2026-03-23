#!/bin/bash
# Run baseline + STTR + STTR+Qwen on random 100-sentence subset (seed=42)
# Usage:
#   bash scripts/run_rand100.sh               # NLLB only
#   bash scripts/run_rand100.sh --with-qwen   # enable Qwen refiner

set -e

SOURCE="data/enzh/rand100_source.txt"
TARGET="data/enzh/rand100_target.txt"
K=5
MODE="tail3"
QWEN_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
QWEN_ARGS=""

for arg in "$@"; do
    case $arg in
        --with-qwen)
            QWEN_ARGS="--qwen-model-path $QWEN_PATH --qwen-gpu 1 --qwen-mode prefix"
            echo "[INFO] Qwen refiner enabled (prefix mode)"
            ;;
    esac
done

if [ ! -f "$SOURCE" ]; then
    echo "Random 100-sentence subset not found. Run: python3 scripts/create_rand100.py"
    exit 1
fi

run_exp() {
    local NAME=$1; shift
    local OUT="outputs/rand100_${NAME}"
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

# 1. Baseline (wait-k only, no refinement)
run_exp "baseline_k${K}" \
    --wait-k $K --beam-size 1 \
    --uncertainty-threshold 999 \
    --uncertainty-mode $MODE

# 2. STTR tau=3.0 (best from prior ablation)
run_exp "sttr_k${K}_tau3.0" \
    --wait-k $K --beam-size 1 \
    --uncertainty-threshold 3.0 \
    --uncertainty-mode $MODE \
    --num-candidates 4 --max-extra-reads 3

# 3. STTR tau=5.0 (lower latency than tau=3.0)
run_exp "sttr_k${K}_tau5.0" \
    --wait-k $K --beam-size 1 \
    --uncertainty-threshold 5.0 \
    --uncertainty-mode $MODE \
    --num-candidates 4 --max-extra-reads 3

# 4. STTR + Qwen (if enabled)
if [ -n "$QWEN_ARGS" ]; then
    run_exp "sttr_k${K}_tau3.0_qwen" \
        --wait-k $K --beam-size 1 \
        --uncertainty-threshold 3.0 \
        --uncertainty-mode $MODE \
        --num-candidates 4 --max-extra-reads 3 \
        $QWEN_ARGS

    run_exp "sttr_k${K}_tau5.0_qwen" \
        --wait-k $K --beam-size 1 \
        --uncertainty-threshold 5.0 \
        --uncertainty-mode $MODE \
        --num-candidates 4 --max-extra-reads 3 \
        $QWEN_ARGS
fi

echo ""
echo "============================================"
echo "All experiments complete! BLEU/AL Summary:"
echo "============================================"
for dir in outputs/rand100_*/; do
    if [ -f "${dir}scores" ]; then
        python3 -c "
import re, sys
name = '$(basename $dir)'
txt = open('${dir}scores').read()
nums = re.findall(r'[\d]+\.[\d]+', txt)
bleu = nums[0] if nums else 'N/A'
al   = nums[2] if len(nums)>2 else 'N/A'
print(f'  {name:<45}  BLEU={bleu:<8}  AL={al}')
"
    fi
done

echo ""
echo "Now running COMET evaluation..."
python3 scripts/eval_comet.py --all-rand100 --output outputs/rand100_comet_results.json
