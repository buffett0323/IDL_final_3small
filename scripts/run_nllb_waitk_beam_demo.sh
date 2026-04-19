#!/usr/bin/env bash
# Strict NLLB simultaneous baseline: wait-k + beam decode only on the current prefix.
# Disables STTR extras (no read-more, no entropy-triggered path that uses multi-candidate LCP).
# Uses sttr_enzh_agent.py with a huge uncertainty threshold + zero extra reads.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
OUT="${REPO_ROOT}/outputs/demo_nllb_waitk_beam"
mkdir -p "${OUT}"

: "${BEAM_SIZE:=8}"
echo "[demo] NLLB wait-k + beam (no STTR read-more/LCP path) beam=${BEAM_SIZE} -> ${OUT}"

simuleval \
  --agent "${REPO_ROOT}/agents/sttr_enzh_agent.py" \
  --source "${REPO_ROOT}/data/enzh/test_source_5.txt" \
  --target "${REPO_ROOT}/data/enzh/test_target_5.txt" \
  --source-lang eng_Latn \
  --target-lang zho_Hans \
  --model-name facebook/nllb-200-distilled-600M \
  --wait-k 5 \
  --beam-size "${BEAM_SIZE}" \
  --uncertainty-threshold 1e9 \
  --max-extra-reads 0 \
  --output "${OUT}"

echo "[demo] Done."
