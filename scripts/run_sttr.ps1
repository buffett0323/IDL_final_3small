# Run STTR experiments
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_sttr.ps1

$SOURCE = "data/wmt/wmt14_source.txt"
$TARGET = "data/wmt/wmt14_target.txt"
$K = 5  # wait-k value

# --- Method B: Always-refine upper bound ---
$outDir = "outputs/always_refine_k$K"
if (-not (Test-Path "$outDir/instances.log")) {
    Write-Host "============================================"
    Write-Host "Method B: Always-refine (k=$K)"
    Write-Host "============================================"
    if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
    simuleval `
        --agent agents/sttr_agent.py `
        --source $SOURCE `
        --target $TARGET `
        --wait-k $K `
        --always-refine `
        --num-candidates 4 `
        --output $outDir
} else {
    Write-Host "=== Skipping always-refine k=$K (already done) ==="
}

# --- Method C: STTR with various tau thresholds ---
foreach ($tau in 1.0, 1.5, 2.0, 2.5, 3.0) {
    $outDir = "outputs/sttr_k${K}_tau${tau}"
    if (-not (Test-Path "$outDir/instances.log")) {
        Write-Host "============================================"
        Write-Host "STTR: k=$K, tau=$tau"
        Write-Host "============================================"
        if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
        simuleval `
            --agent agents/sttr_agent.py `
            --source $SOURCE `
            --target $TARGET `
            --wait-k $K `
            --uncertainty-threshold $tau `
            --num-candidates 4 `
            --output $outDir
    } else {
        Write-Host "=== Skipping STTR k=$K tau=$tau (already done) ==="
    }
}

# Score all results
Write-Host "============================================"
Write-Host "Scoring all results..."
Write-Host "============================================"
python scripts/score_baselines.py --output-dirs `
    outputs/baseline_k$K `
    outputs/always_refine_k$K `
    outputs/sttr_k${K}_tau1.0 `
    outputs/sttr_k${K}_tau1.5 `
    outputs/sttr_k${K}_tau2.0 `
    outputs/sttr_k${K}_tau2.5 `
    outputs/sttr_k${K}_tau3.0
