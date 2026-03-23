#!/bin/bash
# Run all missing full-dataset (1997 sentences) experiments
# GPU0: NLLB,  GPU1: Qwen (when needed)
set -e

SOURCE="data/enzh/wmt19_source.txt"
TARGET="data/enzh/wmt19_target.txt"
K=5
MODE="tail3"
QWEN_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"

run_exp() {
    local NAME=$1; shift
    local OUT="outputs/${NAME}"
    if [ -d "$OUT" ] && [ -f "${OUT}/scores" ]; then
        echo "[SKIP] $NAME already has scores"
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

# ---- 1. tau=6.0 (no Qwen, ~35 min) ----
run_exp "enzh_v2_tau6.0" \
    --wait-k $K --beam-size 1 \
    --uncertainty-threshold 6.0 \
    --uncertainty-mode $MODE \
    --num-candidates 4 --max-extra-reads 3

# ---- 2. tau=7.0 (no Qwen, ~35 min) ----
run_exp "enzh_v2_tau7.0" \
    --wait-k $K --beam-size 1 \
    --uncertainty-threshold 7.0 \
    --uncertainty-mode $MODE \
    --num-candidates 4 --max-extra-reads 3

# ---- 3. tau=3.0 + Qwen prefix mode (new fix, ~2-3 hrs) ----
run_exp "enzh_v2_qwen_prefix_tau3.0" \
    --wait-k $K --beam-size 1 \
    --uncertainty-threshold 3.0 \
    --uncertainty-mode $MODE \
    --num-candidates 4 --max-extra-reads 3 \
    --qwen-model-path "$QWEN_PATH" \
    --qwen-gpu 1 \
    --qwen-mode prefix

# ---- 4. tau=5.0 + Qwen prefix mode (new fix, ~2-3 hrs) ----
run_exp "enzh_v2_qwen_prefix_tau5.0" \
    --wait-k $K --beam-size 1 \
    --uncertainty-threshold 5.0 \
    --uncertainty-mode $MODE \
    --num-candidates 4 --max-extra-reads 3 \
    --qwen-model-path "$QWEN_PATH" \
    --qwen-gpu 1 \
    --qwen-mode prefix

echo ""
echo "============================================"
echo "All missing experiments complete!"
echo "============================================"
python3 << 'PYEOF'
import os, re
base = 'outputs'
for d in sorted(os.listdir(base)):
    if not d.startswith('enzh_v2'): continue
    s = os.path.join(base, d, 'scores')
    if os.path.exists(s):
        txt = open(s).read()
        nums = re.findall(r'[0-9]+\.[0-9]+', txt)
        bleu = nums[0] if nums else 'N/A'
        al   = nums[2] if len(nums)>2 else 'N/A'
        print(f"  {d:<40}  BLEU={bleu:<8} AL={al}")
PYEOF
