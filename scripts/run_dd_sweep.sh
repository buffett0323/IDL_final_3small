#!/bin/bash
# run_dd_sweep.sh — Baseline + DD-gate threshold sweep on rand100
#
# Runs simuleval for:
#   1. Baseline (wait-k=5, no DD gate, uncertainty-threshold=999 to disable entropy gate)
#   2. DD gate with tau ∈ {0.01, 0.02, 0.03, 0.05, 0.08, 0.10}
#
# Key fix (v2): The agent now loads the FULL source sentence from --source at
#   startup and passes it to compute_dd_score as oracle_source_words.  Without
#   this, all K truncation futures are identical (= observed prefix) → JS = 0.
#
# DD trace (dd_trace.jsonl) is written automatically whenever --dd-gate AND
#   --output are both given. No extra flag needed.
#
# Usage:
#   # NLLB baseline + sweep (default):
#   bash scripts/run_dd_sweep.sh
#
#   # Qwen4B base model:
#   bash scripts/run_dd_sweep.sh --causal-lm
#
#   # Single tau quick test:
#   bash scripts/run_dd_sweep.sh --single-tau 0.05
#
#   # Fast mode (K=2, n_steps=1, single-step JS only):
#   bash scripts/run_dd_sweep.sh --fast
#
#   # Force re-run even if partial results exist:
#   bash scripts/run_dd_sweep.sh --force
#
set -e
cd "$(dirname "$0")/.."   # repo root

# ── Parse args ────────────────────────────────────────────────────────────────
CAUSAL_FLAG=""
MODEL="facebook/nllb-200-distilled-600M"
SINGLE_TAU=""
DD_STEPS=3
DD_K=4
WAIT_K=5
DEVICE="cuda:0"
FORCE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --causal-lm)
            CAUSAL_FLAG="--causal-lm"
            MODEL="/data/user_data/haolingp/models/Qwen3-4B-Base"
            shift ;;
        --single-tau)
            SINGLE_TAU="$2"; shift 2 ;;
        --dd-steps)
            DD_STEPS="$2"; shift 2 ;;
        --dd-k)
            DD_K="$2"; shift 2 ;;
        --fast)
            DD_K=2; DD_STEPS=1; shift ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --force)
            FORCE="1"; shift ;;
        *)
            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Shared config ─────────────────────────────────────────────────────────────
SRC="data/enzh/rand100_source.txt"
TGT="data/enzh/rand100_target.txt"
TAU_LIST=(0.01 0.02 0.03 0.05 0.08 0.10)

if [[ -n "$SINGLE_TAU" ]]; then
    TAU_LIST=("$SINGLE_TAU")
fi

SUFFIX=""
if [[ -n "$CAUSAL_FLAG" ]]; then SUFFIX="_qwen4b"; fi

echo "════════════════════════════════════════════════════════════"
echo "DD Gate Sweep on rand100  (oracle truncation futures, batched)"
echo "  Model     : $MODEL"
echo "  DD steps  : $DD_STEPS  (policy score: avg_js_firstN)"
echo "  DD K      : $DD_K"
echo "  Taus      : ${TAU_LIST[*]}"
echo "  Device    : $DEVICE"
echo "════════════════════════════════════════════════════════════"
date

# ── Helper: clean partial run ─────────────────────────────────────────────────
# A run is "partial" if its directory exists but has no scores file.
# We clean it so simuleval starts fresh (its own config.yaml / instances.log
# are created at start and would otherwise conflict).
clean_partial() {
    local dir="$1"
    if [[ -d "$dir" ]] && [[ ! -f "$dir/scores" ]]; then
        echo "[Cleanup] Removing partial result at $dir"
        rm -rf "$dir"
    fi
}

# ── 1. Baseline ───────────────────────────────────────────────────────────────
BASELINE_OUT="outputs/dd_sweep${SUFFIX}_baseline"

if [[ -n "$FORCE" ]]; then
    echo "[Baseline] --force: removing existing $BASELINE_OUT"
    rm -rf "$BASELINE_OUT"
fi

if [[ -f "$BASELINE_OUT/scores" ]]; then
    echo "[Baseline] Already complete at $BASELINE_OUT, skipping."
    echo "  Scores:"; cat "$BASELINE_OUT/scores"
else
    clean_partial "$BASELINE_OUT"
    echo ""
    echo "── Running baseline ──────────────────────────────────────"
    mkdir -p "$BASELINE_OUT"
    CUDA_VISIBLE_DEVICES="${DEVICE##*:}" simuleval \
        --agent agents/sttr_enzh_agent.py \
        --source "$SRC" \
        --target "$TGT" \
        --source-lang eng_Latn \
        --target-lang zho_Hans \
        --wait-k "$WAIT_K" \
        --model-name "$MODEL" \
        $CAUSAL_FLAG \
        --uncertainty-threshold 999 \
        --device "$DEVICE" \
        --output "$BASELINE_OUT" \
        2>&1 | tee "$BASELINE_OUT/run.log"
    echo "[Baseline] Done."
    [[ -f "$BASELINE_OUT/scores" ]] && echo "Scores:" && cat "$BASELINE_OUT/scores"
fi

# ── 2. DD gate sweep ──────────────────────────────────────────────────────────
for TAU in "${TAU_LIST[@]}"; do
    OUT="outputs/dd_sweep${SUFFIX}_tau${TAU}"

    if [[ -n "$FORCE" ]]; then
        echo "[DD tau=$TAU] --force: removing existing $OUT"
        rm -rf "$OUT"
    fi

    if [[ -f "$OUT/scores" ]]; then
        echo "[DD tau=$TAU] Already complete at $OUT, skipping."
        echo "  Scores:"; cat "$OUT/scores"
        continue
    fi

    # Clean any partial run (no scores file = broken)
    clean_partial "$OUT"

    echo ""
    echo "── DD gate  tau=$TAU  (K=$DD_K  steps=$DD_STEPS) ──────────"
    mkdir -p "$OUT"
    CUDA_VISIBLE_DEVICES="${DEVICE##*:}" simuleval \
        --agent agents/sttr_enzh_agent.py \
        --source "$SRC" \
        --target "$TGT" \
        --source-lang eng_Latn \
        --target-lang zho_Hans \
        --wait-k "$WAIT_K" \
        --model-name "$MODEL" \
        $CAUSAL_FLAG \
        --uncertainty-threshold 999 \
        --device "$DEVICE" \
        --dd-gate \
        --dd-tau "$TAU" \
        --dd-futures-k "$DD_K" \
        --dd-steps "$DD_STEPS" \
        --output "$OUT" \
        2>&1 | tee "$OUT/run.log"
    echo "[DD tau=$TAU] Done."
    [[ -f "$OUT/scores" ]] && echo "Scores:" && cat "$OUT/scores"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "All runs done. Generating comparison report..."
python scripts/analyze_dd_results.py \
    --suffix "$SUFFIX" \
    --taus "${TAU_LIST[@]}" \
    --dd-steps "$DD_STEPS" \
    --dd-k "$DD_K"
echo "Report written."
date
