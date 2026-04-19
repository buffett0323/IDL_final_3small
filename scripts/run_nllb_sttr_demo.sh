#!/usr/bin/env bash
# NLLB STTR-v2 (no DD): uncertainty + read-more + occasional multi-candidate LCP.
# This is NOT the strict wait-k+beam baseline — use run_nllb_waitk_beam_demo.sh for that.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
OUT="${REPO_ROOT}/outputs/demo_nllb_sttr"
mkdir -p "${OUT}"

echo "[demo] NLLB STTR-v2 (no DD) -> ${OUT}"

simuleval \
  --agent "${REPO_ROOT}/agents/sttr_enzh_agent.py" \
  --source "${REPO_ROOT}/data/enzh/test_source_5.txt" \
  --target "${REPO_ROOT}/data/enzh/test_target_5.txt" \
  --source-lang eng_Latn \
  --target-lang zho_Hans \
  --model-name facebook/nllb-200-distilled-600M \
  --wait-k 5 \
  --uncertainty-threshold 1.5 \
  --output "${OUT}"

echo "[demo] Done. Optional: --trace-refinement → refine_trace.jsonl"
