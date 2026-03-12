#!/bin/bash
# Run wait-k baseline experiments for k in {3, 5, 7, 9}
# Usage: bash scripts/run_baseline.sh

SOURCE="data/wmt/wmt14_source.txt"
TARGET="data/wmt/wmt14_target.txt"

for k in 3 5 7 9; do
    echo "============================================"
    echo "Running wait-k=$k baseline"
    echo "============================================"
    simuleval \
        --agent agents/waitk_agent.py \
        --source "$SOURCE" \
        --target "$TARGET" \
        --wait-k "$k" \
        --output "outputs/baseline_k${k}/"
    echo ""
done

echo "All baselines complete!"
echo "Results are in outputs/baseline_k{3,5,7,9}/"
