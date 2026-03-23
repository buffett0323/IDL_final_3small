# STTR: Selective Test-Time Reasoning for Simultaneous Translation

CMU 11-785 Introduction to Deep Learning — Final Project

## Overview

STTR extends the standard **wait-k** simultaneous translation policy with **uncertainty-gated selective compute**. The key idea: most translations are easy (the model is confident), so we only spend extra compute on the hard ones.

### Current mainline: STTR-v2 (EN->ZH)

The project has evolved from beam-search refinement (v1) to a more principled approach:

**Why the change from beam-refine to read-more + LCP + triggered Qwen?**
- Beam-search refinement on MarianMT showed only +0.2 BLEU for 4x compute cost
- Read-more is a zero-cost way to reduce uncertainty (just wait for more source context)
- LCP (Longest Common Prefix) commit is more principled than picking a single beam output — it only emits characters that ALL candidates agree on
- Qwen3 reranking is reserved for the hardest cases (no stable LCP + source finished), keeping average latency low

**STTR-v2 Pipeline:**
1. Read k source words, then start translating (wait-k policy)
2. Generate a draft translation with greedy decoding (beam=1)
3. Compute token-level uncertainty (entropy / margin)
4. If uncertain and source available: **read more** source words (up to 3 extra)
5. If still uncertain or source finished: generate K candidates, commit their **Longest Common Prefix (LCP)**
6. If no stable LCP and source finished: trigger **Qwen3 reranker** (optional, hard cases only)

**Language pair:** English to Chinese (EN->ZH)
**Base model:** facebook/nllb-200-distilled-600M (~600M params)
**Reranker:** Qwen3-30B-A3B-Instruct (FP8, triggered on hard cases only)
**Dataset:** WMT19 En-Zh newstest (1997 sentences)
**Evaluation:** SimulEval framework — BLEU (quality) and Average Lagging / AL (latency)

### Chinese emission granularity

Chinese text has no whitespace word boundaries. We use **character-level emission**: each CJK character is one emission unit. Mixed content (numbers, Latin) is grouped as single units. The reference file is pre-segmented with spaces between characters so SimulEval counts them correctly for latency metrics. BLEU scoring is unaffected because sacrebleu tokenizes Chinese by character.

## Methods

### STTR-v2 (EN->ZH, current)

| Method | Description |
|--------|-------------|
| **Baseline (wait-k, beam=1)** | Standard wait-k with greedy decoding, character-level emission |
| **STTR-v2 (read-more + LCP)** | Uncertainty-gated: draft -> read-more -> LCP commit |
| **STTR-v2 + Qwen** | Same as above, plus triggered Qwen3 rerank on hard cases |
| **Always-LCP (upper bound)** | Always generates K candidates and commits LCP |

Uncertainty modes: `mean` (whole-sequence entropy), `last` (last-token entropy), `tail3` (tail-3 token entropy), `margin` (1 - top1-top2 probability gap).

### STTR-v1 (EN->DE, legacy)

| Method | Description |
|--------|-------------|
| **Baseline (wait-k, beam=1)** | Standard wait-k with greedy decoding |
| **Baseline (wait-k, beam=8)** | Compute-matched baseline — always uses 8 beams |
| **STTR-v1** | Uncertainty-gated: draft with beam=1, refine with beam=8 only when entropy > tau |

## Preliminary Results (EN->DE, legacy)

### Wait-k Baselines (beam=1, greedy)

| wait-k | BLEU | AL |
|--------|------|----|
| k=3 | 14.17 | 2.50 |
| k=5 | 16.72 | 4.14 |
| k=7 | 18.43 | 6.08 |
| k=9 | 19.63 | 8.03 |

### STTR-v1 vs Compute-Matched Baseline (k=5)

| Method | BLEU | AL | Wall Time |
|--------|------|----|-----------|
| Baseline k=5, beam=1 | 16.72 | 4.14 | ~30 min |
| Baseline k=5, beam=8 | 16.92 | 4.16 | 2:08:26 |
| STTR k=5, tau=2.0 | 16.81 | 4.15 | 2:28:31 |

### Why we moved on from beam-refine

- Beam=8 provides minimal improvement over beam=1 (+0.2 BLEU) on opus-mt, suggesting the model is already well-calibrated for greedy decoding.
- The marginal benefit of beam search refinement is small — motivating the switch to read-more + LCP + triggered large-model rerank.

## Project Structure

```
agents/
  waitk_agent.py       # Wait-k baseline SimulEval agent (EN->DE legacy)
  sttr_agent.py        # STTR-v1 agent with beam refinement (EN->DE legacy)
  sttr_enzh_agent.py   # STTR-v2 agent: NLLB EN->ZH + read-more + LCP + Qwen rerank
  model_utils.py       # Model loading, Chinese char splitting, language code handling
scripts/
  download_wmt.py      # Download WMT14 En-De test sets
  download_enzh_data.py # Download WMT19 En-Zh test sets (char-segmented)
  run_enzh_smoke.sh    # Quick 5-sentence EN->ZH smoke test
  run_enzh_full.sh     # Full WMT19 EN->ZH experiment (optional --with-qwen)
  score_baselines.py   # Score output directories (BLEU + AL)
  run_baseline.sh      # Legacy: wait-k baselines for k in {3,5,7,9}
data/
  enzh/                # EN->ZH test data (char-segmented references)
  wmt/                 # Legacy EN->DE WMT14 data
tests/
  test_enzh_agent_logic.py    # 21 tests: Chinese splitting, LCP, gate logic
  test_sttr_agent_logic.py    # Legacy STTR-v1 gate logic tests
  test_refinement_analysis.py # Analysis utility tests
outputs/
  enzh_*/              # EN->ZH experiment results
  baseline_k*/         # Legacy EN->DE results
```

## Setup

```bash
# Install PyTorch first (see https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

### Model downloads

Models are cached to `/data/user_data/haolingp/models` by default. The NLLB model (~600M) downloads automatically on first run. To pre-download:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
  AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='/data/user_data/haolingp/models'); \
  AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='/data/user_data/haolingp/models')"
```

Qwen3-30B-A3B-Instruct-FP8 should already be at `/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8/`.

## Running Experiments (EN->ZH)

### Smoke test (5 sentences, ~15s)

```bash
bash scripts/run_enzh_smoke.sh
```

### Full experiment (WMT19, ~2000 sentences)

```bash
# Download WMT19 En-Zh data first
python scripts/download_enzh_data.py

# Run tau sweep (NLLB only, GPU0)
bash scripts/run_enzh_full.sh

# Run with Qwen reranker enabled (GPU0 + GPU1)
bash scripts/run_enzh_full.sh --with-qwen
```

### Unit tests

```bash
python -m pytest tests/ -v
```

## Legacy experiments (EN->DE)

```bash
# Download WMT14 En-De data
python scripts/download_wmt.py

# Wait-k baselines
bash scripts/run_baseline.sh
```

## GPU Assignment

| GPU | Model | Purpose |
|-----|-------|---------|
| GPU0 | NLLB-200-distilled-600M | Base simultaneous decoding + candidate generation |
| GPU1 | Qwen3-30B-A3B-Instruct-FP8 | Triggered reranking on hard cases (optional) |

## Next Steps

- Run full WMT19 En-Zh tau sweep and find optimal threshold
- Pareto frontier plot: BLEU vs AL across methods
- Error analysis: do LCP commits correlate with actual hard cases?
- Tune max-extra-reads and num-candidates
- Compare character-level vs subword-level emission
