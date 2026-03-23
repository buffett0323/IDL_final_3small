#!/bin/bash
# Smoke test for EN->ZH STTR-v2 pipeline (5 sentences)
# Usage: bash scripts/run_enzh_smoke.sh

set -e

SOURCE="data/enzh/test_source_5.txt"
TARGET="data/enzh/test_target_5.txt"
K=5
TAU=4.0
MODE="tail3"

# ---- 1. Baseline: wait-k greedy ----
OUT="outputs/enzh_baseline_k${K}"
echo "============================================"
echo "EN->ZH Baseline: wait-k=${K}, beam=1"
echo "============================================"
rm -rf "$OUT"
simuleval \
    --agent agents/sttr_enzh_agent.py \
    --source "$SOURCE" \
    --target "$TARGET" \
    --source-lang eng_Latn \
    --target-lang zho_Hans \
    --wait-k "$K" \
    --beam-size 1 \
    --uncertainty-threshold 999 \
    --uncertainty-mode "$MODE" \
    --base-gpu 0 \
    --trace-refinement \
    --output "$OUT"
echo ""

# ---- 2. STTR-v2: read-more + LCP ----
OUT="outputs/enzh_sttr_k${K}_tau${TAU}"
echo "============================================"
echo "EN->ZH STTR-v2: k=${K}, tau=${TAU}, mode=${MODE}"
echo "============================================"
rm -rf "$OUT"
simuleval \
    --agent agents/sttr_enzh_agent.py \
    --source "$SOURCE" \
    --target "$TARGET" \
    --source-lang eng_Latn \
    --target-lang zho_Hans \
    --wait-k "$K" \
    --beam-size 1 \
    --uncertainty-threshold "$TAU" \
    --uncertainty-mode "$MODE" \
    --num-candidates 4 \
    --max-extra-reads 3 \
    --base-gpu 0 \
    --trace-refinement \
    --output "$OUT"
echo ""

echo "============================================"
echo "Smoke test complete!"
echo "============================================"
echo "Baseline: outputs/enzh_baseline_k${K}/scores"
echo "STTR-v2:  outputs/enzh_sttr_k${K}_tau${TAU}/scores"
