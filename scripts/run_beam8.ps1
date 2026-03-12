# Run beam-8 baseline and STTR experiments for fair comparison
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_beam8.ps1

$SOURCE = "data/wmt/wmt14_source.txt"
$TARGET = "data/wmt/wmt14_target.txt"
$K = 5
$timingFile = "outputs/timing.txt"

# Ensure outputs dir exists
if (-not (Test-Path "outputs")) { New-Item -ItemType Directory -Path "outputs" }

# --- Baseline: wait-k with beam=8 (compute-matched upper bound) ---
$outDir = "outputs/baseline_k${K}_beam8"
if (-not (Test-Path "$outDir/instances.log")) {
    Write-Host "============================================"
    Write-Host "Baseline: wait-k=$K, beam=8"
    Write-Host "============================================"
    if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    simuleval `
        --agent agents/waitk_agent.py `
        --source $SOURCE `
        --target $TARGET `
        --wait-k $K `
        --beam-size 8 `
        --output $outDir
    $sw.Stop()
    $elapsed = $sw.Elapsed.ToString("hh\:mm\:ss")
    $line = "baseline_k${K}_beam8: $elapsed"
    Write-Host $line
    Add-Content -Path $timingFile -Value $line
} else {
    Write-Host "=== Skipping baseline k=$K beam=8 (already done) ==="
}

# --- STTR: wait-k with beam=1 draft, beam=8 refine, tau=2.0 ---
$outDir = "outputs/sttr_k${K}_tau2.0"
if (-not (Test-Path "$outDir/instances.log")) {
    Write-Host "============================================"
    Write-Host "STTR: k=$K, tau=2.0, candidates=4 (refine beam=8)"
    Write-Host "============================================"
    if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    simuleval `
        --agent agents/sttr_agent.py `
        --source $SOURCE `
        --target $TARGET `
        --wait-k $K `
        --uncertainty-threshold 2.0 `
        --num-candidates 4 `
        --output $outDir
    $sw.Stop()
    $elapsed = $sw.Elapsed.ToString("hh\:mm\:ss")
    $line = "sttr_k${K}_tau2.0: $elapsed"
    Write-Host $line
    Add-Content -Path $timingFile -Value $line
} else {
    Write-Host "=== Skipping STTR k=$K tau=2.0 (already done) ==="
}

# Score results
Write-Host "============================================"
Write-Host "Scoring results..."
Write-Host "============================================"
python scripts/score_baselines.py --output-dirs `
    outputs/baseline_k${K} `
    outputs/baseline_k${K}_beam8 `
    outputs/sttr_k${K}_tau2.0

Write-Host ""
Write-Host "============================================"
Write-Host "Timing results:"
Write-Host "============================================"
Get-Content $timingFile
