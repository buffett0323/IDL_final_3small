# Run wait-k baseline experiments for k in {3, 5, 7, 9}
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1

$SOURCE = "data/wmt/wmt14_source.txt"
$TARGET = "data/wmt/wmt14_target.txt"

foreach ($k in 3, 5, 7, 9) {
    $outDir = "outputs/baseline_k$k"
    $logFile = "$outDir/instances.log"

    if (Test-Path $logFile) {
        Write-Host "=== Skipping wait-k=$k (instances.log already exists) ==="
        continue
    }

    Write-Host "============================================"
    Write-Host "Running wait-k=$k baseline"
    Write-Host "============================================"

    if (Test-Path $outDir) {
        Remove-Item -Recurse -Force $outDir
    }

    simuleval `
        --agent agents/waitk_agent.py `
        --source $SOURCE `
        --target $TARGET `
        --wait-k $k `
        --beam-size 1 `
        --output $outDir

    Write-Host ""
}

# Score all results
Write-Host "============================================"
Write-Host "Scoring all baselines..."
Write-Host "============================================"
python scripts/score_baselines.py

Write-Host ""
Write-Host "All baselines complete!"
