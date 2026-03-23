#!/bin/bash
# 100-case ablation test: compare baseline vs STTR variants vs Qwen-fixed
# Uses wmt19_source_100.txt / wmt19_target_100.txt
# Usage:
#   bash scripts/run_100case_test.sh               # NLLB only (GPU0)
#   bash scripts/run_100case_test.sh --with-qwen   # Enable Qwen refiner (GPU0+GPU1)
#   bash scripts/run_100case_test.sh --quick       # Only baseline + best tau (faster)

set -e

SOURCE="data/enzh/wmt19_source_100.txt"
TARGET="data/enzh/wmt19_target_100.txt"
K=5
MODE="tail3"
QWEN_PATH="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
QWEN_ARGS=""
QUICK=0

for arg in "$@"; do
    case $arg in
        --with-qwen) QWEN_ARGS="--qwen-model-path $QWEN_PATH --qwen-gpu 1"; echo "[INFO] Qwen enabled" ;;
        --quick) QUICK=1; echo "[INFO] Quick mode: baseline + tau=5.0 only" ;;
    esac
done

if [ ! -f "$SOURCE" ]; then
    echo "100-sentence subset not found. Downloading WMT19 En-Zh ..."
    python scripts/download_enzh_data.py
fi

run_exp() {
    local NAME=$1; shift
    local OUT="outputs/test100_${NAME}"
    echo ""
    echo "============================================"
    echo "Running: $NAME  ->  $OUT"
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
}

# ---- 1. Baseline (tau=999 = never refine) ----
run_exp "baseline_k${K}" \
    --wait-k $K --beam-size 1 \
    --uncertainty-threshold 999 \
    --uncertainty-mode $MODE

if [ $QUICK -eq 1 ]; then
    # Quick mode: only best known tau
    run_exp "sttr_k${K}_tau5.0" \
        --wait-k $K --beam-size 1 \
        --uncertainty-threshold 5.0 \
        --uncertainty-mode $MODE \
        --num-candidates 4 --max-extra-reads 3 \
        $QWEN_ARGS
else
    # ---- 2. STTR tau sweep ----
    for TAU in 3.0 4.0 5.0 6.0 7.0; do
        run_exp "sttr_k${K}_tau${TAU}" \
            --wait-k $K --beam-size 1 \
            --uncertainty-threshold "$TAU" \
            --uncertainty-mode $MODE \
            --num-candidates 4 --max-extra-reads 3 \
            $QWEN_ARGS
    done

    # ---- 3. seq_logprob uncertainty mode (new) ----
    run_exp "sttr_k${K}_seqlogprob_tau0.5" \
        --wait-k $K --beam-size 1 \
        --uncertainty-threshold 0.5 \
        --uncertainty-mode seq_logprob \
        --num-candidates 4 --max-extra-reads 3

    run_exp "sttr_k${K}_seqlogprob_tau1.0" \
        --wait-k $K --beam-size 1 \
        --uncertainty-threshold 1.0 \
        --uncertainty-mode seq_logprob \
        --num-candidates 4 --max-extra-reads 3

    run_exp "sttr_k${K}_seqlogprob_tau2.0" \
        --wait-k $K --beam-size 1 \
        --uncertainty-threshold 2.0 \
        --uncertainty-mode seq_logprob \
        --num-candidates 4 --max-extra-reads 3

    # ---- 4. More candidates (K=8) ----
    run_exp "sttr_k${K}_tau5.0_cand8" \
        --wait-k $K --beam-size 1 \
        --uncertainty-threshold 5.0 \
        --uncertainty-mode $MODE \
        --num-candidates 8 --max-extra-reads 3

    # ---- 5. Qwen with fixed prompt (if enabled) ----
    if [ -n "$QWEN_ARGS" ]; then
        run_exp "sttr_k${K}_tau3.0_qwen_fixed" \
            --wait-k $K --beam-size 1 \
            --uncertainty-threshold 3.0 \
            --uncertainty-mode $MODE \
            --num-candidates 4 --max-extra-reads 3 \
            $QWEN_ARGS

        run_exp "sttr_k${K}_tau5.0_qwen_fixed" \
            --wait-k $K --beam-size 1 \
            --uncertainty-threshold 5.0 \
            --uncertainty-mode $MODE \
            --num-candidates 4 --max-extra-reads 3 \
            $QWEN_ARGS
    fi
fi

# ---- Summary ----
echo ""
echo "============================================"
echo "All experiments complete! Summary:"
echo "============================================"
for dir in outputs/test100_*/; do
    if [ -f "${dir}scores" ]; then
        BLEU=$(grep -oP 'BLEU\s*\|\s*\K[\d.]+' "${dir}scores" 2>/dev/null | head -1 || echo "N/A")
        AL=$(grep -oP 'AL\s*\|\s*\K[\d.]+' "${dir}scores" 2>/dev/null | head -1 || echo "N/A")
        printf "  %-45s  BLEU=%-8s  AL=%s\n" "$(basename $dir)" "$BLEU" "$AL"
    fi
done
