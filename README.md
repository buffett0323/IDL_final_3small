# STTR: Selective Test-Time Reasoning for Simultaneous Translation

CMU 11-785 Introduction to Deep Learning — Final Project

## Overview

STTR extends the standard **wait-k** simultaneous translation policy with **uncertainty-gated refinement**. The key idea: most translations are easy (the model is confident), so we only spend extra compute on the hard ones.

**Pipeline:**
1. Read k source words, then start translating (wait-k policy)
2. Generate a draft translation with greedy decoding (beam=1)
3. Compute token-level entropy as an uncertainty signal
4. If entropy > threshold tau, refine via multi-candidate beam search (beam=8)
5. Otherwise, keep the cheap draft

**Language pair:** English to German (En-De)
**Model:** Helsinki-NLP/opus-mt-en-de (~300M params, MarianMT)
**Dataset:** WMT14 newstest (2737 sentences)
**Evaluation:** SimulEval framework — BLEU (quality) and Average Lagging / AL (latency)

## Methods

| Method | Description |
|--------|-------------|
| **Baseline (wait-k, beam=1)** | Standard wait-k with greedy decoding |
| **Baseline (wait-k, beam=8)** | Compute-matched baseline — always uses 8 beams |
| **Method B (always-refine)** | Always runs refinement at every commit point (upper bound) |
| **Method C (STTR)** | Uncertainty-gated: draft with beam=1, refine with beam=8 only when entropy > tau |

## Preliminary Results

### Wait-k Baselines (beam=1, greedy)

| wait-k | BLEU | AL |
|--------|------|----|
| k=3 | 14.17 | 2.50 |
| k=5 | 16.72 | 4.14 |
| k=7 | 18.43 | 6.08 |
| k=9 | 19.63 | 8.03 |

Higher k = more source context before translating = better BLEU but higher latency. This is the expected quality-latency tradeoff.

### STTR vs Compute-Matched Baseline (k=5)

| Method | BLEU | AL | Wall Time |
|--------|------|----|-----------|
| Baseline k=5, beam=1 | 16.72 | 4.14 | ~30 min |
| Baseline k=5, beam=8 | 16.92 | 4.16 | 2:08:26 |
| STTR k=5, tau=2.0 | 16.81 | 4.15 | 2:28:31 |

### Key Observations

- Beam=8 provides minimal improvement over beam=1 (+0.2 BLEU) on this model, suggesting opus-mt is already well-calibrated for greedy decoding on short prefixes.
- STTR performs between the two baselines in BLEU (16.81) but takes longer due to draft+refine overhead.
- The marginal benefit of beam search refinement is small for this particular model — a larger model (e.g., NLLB-200) where beam search helps more could show a bigger gap.

## Project Structure

```
agents/
  waitk_agent.py       # Wait-k baseline SimulEval agent
  sttr_agent.py        # STTR agent with uncertainty-gated refinement
scripts/
  download_wmt.py      # Download WMT test sets via sacrebleu
  score_baselines.py   # Score output directories (BLEU + AL)
  run_baseline.ps1     # Run wait-k baselines for k in {3,5,7,9}
  run_sttr.ps1         # Run STTR tau sweep {1.0, 1.5, 2.0, 2.5, 3.0}
  run_beam8.ps1        # Run beam=8 baseline vs STTR (fair comparison)
data/wmt/
  wmt14_source.txt     # English source sentences (SimulEval format)
  wmt14_target.txt     # German reference translations
outputs/
  baseline_k*/         # Wait-k baseline results
  sttr_k*_tau*/        # STTR results for various tau
  timing.txt           # Wall-clock execution times
```

## Setup

```bash
# Install PyTorch first (see https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Download WMT14 En-De test set
python scripts/download_wmt.py
```

## Running Experiments

```powershell
# Wait-k baselines (beam=1)
powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1

# STTR vs beam=8 baseline (fair comparison)
powershell -ExecutionPolicy Bypass -File scripts/run_beam8.ps1

# STTR tau sweep
powershell -ExecutionPolicy Bypass -File scripts/run_sttr.ps1

# Score any output directories
python scripts/score_baselines.py --output-dirs outputs/baseline_k5 outputs/sttr_k5_tau2.0
```

## Next Steps

- Try a larger model (NLLB-200-600M) where beam search has more impact
- Run full tau sweep to find optimal threshold
- Run always-refine (Method B) as quality upper bound
- Error analysis: examine which sentences trigger refinement and whether they correlate with actual translation errors
- Pareto frontier plot: BLEU vs AL across all methods
