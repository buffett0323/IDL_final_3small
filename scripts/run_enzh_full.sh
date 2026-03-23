#!/bin/bash
# Full EN->ZH STTR-v2 experiment on WMT19 data
# Prerequisites: python scripts/download_enzh_data.py
# Usage: bash scripts/run_enzh_full.sh [--with-qwen]

set -e

SOURCE="data/enzh/wmt19_source.txt"
TARGET="data/enzh/wmt19_target.txt"
K=5
MODE="tail3"
QWEN_ARGS=""

if [ "$1" = "--with-qwen" ]; then
    QWEN_ARGS="--qwen-model-path /data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8 --qwen-gpu 1"
    echo "[INFO] Qwen reranker enabled"
fi

if [ ! -f "$SOURCE" ]; then
    echo "Data not found. Downloading WMT19 En-Zh ..."
    python scripts/download_enzh_data.py
fi

# ---- Baseline ----
OUT="outputs/enzh_full_baseline_k${K}"
echo "============================================"
echo "Baseline: wait-k=${K}, beam=1"
echo "============================================"
rm -rf "$OUT"
simuleval \
    --agent agents/sttr_enzh_agent.py \
    --source "$SOURCE" --target "$TARGET" \
    --source-lang eng_Latn --target-lang zho_Hans \
    --wait-k "$K" --beam-size 1 \
    --uncertainty-threshold 999 \
    --uncertainty-mode "$MODE" \
    --base-gpu 0 --trace-refinement \
    --output "$OUT"
echo ""

# ---- STTR-v2 tau sweep ----
for TAU in 2.0 3.0 4.0 5.0 6.0; do
    OUT="outputs/enzh_full_sttr_k${K}_tau${TAU}"
    echo "============================================"
    echo "STTR-v2: k=${K}, tau=${TAU}, mode=${MODE}"
    echo "============================================"
    rm -rf "$OUT"
    simuleval \
        --agent agents/sttr_enzh_agent.py \
        --source "$SOURCE" --target "$TARGET" \
        --source-lang eng_Latn --target-lang zho_Hans \
        --wait-k "$K" --beam-size 1 \
        --uncertainty-threshold "$TAU" \
        --uncertainty-mode "$MODE" \
        --num-candidates 4 --max-extra-reads 3 \
        --base-gpu 0 --trace-refinement \
        $QWEN_ARGS \
        --output "$OUT"
    echo ""
done

echo "============================================"
echo "All experiments complete!"
echo "============================================"
