#!/usr/bin/env bash
# =====================================================================
# COMET scoring wrapper. Calls scripts/comet_score_all.py.
#
# Prerequisites:
#   pip install unbabel-comet
#
# Usage:
#   bash scripts/run_comet_scoring.sh                     # score all
#   bash scripts/run_comet_scoring.sh outputs/wmt19_qwen  # one subtree
#   sbatch scripts/run_comet_scoring.sbatch               # via SLURM
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check COMET installed
if ! python3 -c "import comet" 2>/dev/null; then
    echo "[COMET] Not installed. Run:  pip install unbabel-comet"
    exit 1
fi

SEARCH="${1:-${REPO_ROOT}/outputs}"
python3 "${REPO_ROOT}/scripts/comet_score_all.py" --search-root "${SEARCH}" --repo-root "${REPO_ROOT}"
