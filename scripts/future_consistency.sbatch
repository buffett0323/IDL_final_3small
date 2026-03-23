#!/bin/bash
#SBATCH --job-name=future_consistency
#SBATCH --output=/data/user_data/haolingp/IDL_final_3small/outputs/rand100_future_consistency/slurm_%j_%t.out
#SBATCH --error=/data/user_data/haolingp/IDL_final_3small/outputs/rand100_future_consistency/slurm_%j_%t.err
#SBATCH --time=06:00:00
#
# GPU strategy:
#   We run a COMPARISON experiment: truncation vs lm_sample futures.
#   --compare-modes runs both passes sequentially on 2 GPUs (shard 0 = GPU 0,
#   shard 1 = GPU 1).  Each mode writes to its own subdirectory.
#   A comparison report is written to outputs/rand100_future_consistency/comparison_report.md.
#
#   truncation pass:   fast  (~30 min on 2 GPUs)
#   lm_sample pass:    slower (~90 min on 2 GPUs, temperature sampling overhead)
#
#   To run only one mode without comparison, remove --compare-modes and set
#   --future-mode truncation   (or lm_sample).
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export OUTPUT="$REPO_ROOT/outputs/rand100_future_consistency"

# ── environment ───────────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate base 2>/dev/null || true
fi

# ── config ────────────────────────────────────────────────────────────────────
MODEL="/data/user_data/haolingp/models/Qwen3-4B-Base"
CAUSAL_FLAG="--causal-lm"
# For NLLB (truncation-only), comment the two lines above and uncomment:
# MODEL="facebook/nllb-200-distilled-600M"
# CAUSAL_FLAG=""

NUM_SHARDS=2
K=4
PREFIX_WORDS=5
CONT_LEN=8
LM_TEMP=1.2
LM_TOP_P=0.9
LM_MAX_WORDS=12
# OUTPUT is already set above to absolute path

# Decision thresholds (all configurable)
COMMIT_JS=0.05
READ_JS=0.20
OVERLAP_THRESH=0.5
COMMIT_LCP=2
READ_EDIT=0.80
COMMIT_AGR=0.50

mkdir -p "$OUTPUT"

echo "[sbatch] Starting future-consistency COMPARISON experiment (truncation vs lm_sample)"
echo "[sbatch] Model: $MODEL"
echo "[sbatch] Output: $OUTPUT"
date

# ── run comparison across shards ─────────────────────────────────────────────
# Each shard gets --compare-modes, so it independently runs both passes and
# writes truncation/results_shardN.jsonl + lm_sample/results_shardN.jsonl.
# Shard 0 also produces the merged comparison report.

PIDS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    GPU_ID=$SHARD_ID
    echo "[sbatch] Launching shard ${SHARD_ID} on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python scripts/run_future_consistency.py \
        --model-name "$MODEL" \
        $CAUSAL_FLAG \
        --device cuda:0 \
        --K "$K" \
        --prefix-words "$PREFIX_WORDS" \
        --cont-len "$CONT_LEN" \
        --top-k-overlap 10 \
        --lm-temperature "$LM_TEMP" \
        --lm-top-p "$LM_TOP_P" \
        --lm-max-new-words "$LM_MAX_WORDS" \
        --output "$OUTPUT" \
        --commit-js "$COMMIT_JS" \
        --read-js "$READ_JS" \
        --overlap-thresh "$OVERLAP_THRESH" \
        --commit-lcp "$COMMIT_LCP" \
        --read-edit "$READ_EDIT" \
        --commit-agreement "$COMMIT_AGR" \
        --shard-id "$SHARD_ID" \
        --num-shards "$NUM_SHARDS" \
        --compare-modes \
        --verbose-example-id 0 \
        --resume \
        >> "${OUTPUT}/shard${SHARD_ID}.log" 2>&1 &
    PIDS+=($!)
done

echo "[sbatch] Waiting for ${#PIDS[@]} shards ..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[sbatch] WARNING: A shard failed (pid=$pid)"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "[sbatch] $FAILED shard(s) failed. Check shard*.log files."
    exit 1
fi

echo "[sbatch] All shards done."

# Quick console summary of both modes
python3 - << 'PYEOF'
import json, sys
from pathlib import Path
from statistics import mean

out = Path("outputs/rand100_future_consistency")

def load_mode(mode):
    mode_dir = out / mode
    files = sorted(mode_dir.glob("results_shard*.jsonl"))
    if not files:
        f = mode_dir / "results.jsonl"
        files = [f] if f.exists() else []
    records = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try: records.append(json.loads(line))
                except: pass
    return records

for mode in ["truncation", "lm_sample"]:
    recs = load_mode(mode)
    if not recs:
        print(f"[{mode}] No results found.")
        continue
    n = len(recs)
    js_vals = [r["distribution_divergence"]["avg_js"] for r in recs]
    dd_dec  = [r["distribution_divergence"]["decision"] for r in recs]
    nested  = [r["distribution_divergence"].get("future_diversity",{}).get("nested_prefix_rate",1) for r in recs]
    lcp_len = [r["semantic_lcp"]["literal_lcp_len"] for r in recs]
    sl_dec  = [r["semantic_lcp"]["decision"] for r in recs]
    print(f"\n{'='*60}")
    print(f"MODE: {mode.upper():12}  ({n} examples)")
    print(f"{'='*60}")
    print(f"  Future diversity  nested_rate={mean(nested):.2f}")
    print(f"  Avg JS divergence {mean(js_vals):.4f}")
    print(f"  DD decisions      COMMIT={dd_dec.count('COMMIT')} BORDERLINE={dd_dec.count('BORDERLINE')} READ={dd_dec.count('READ')}")
    print(f"  Avg LCP len       {mean(lcp_len):.2f} chars")
    print(f"  LCP decisions     COMMIT={sl_dec.count('COMMIT')} BORDERLINE={sl_dec.count('BORDERLINE')} READ={sl_dec.count('READ')}")

print(f"\nComparison report → {out}/comparison_report.md")
PYEOF

echo "[sbatch] Experiment complete."
date
