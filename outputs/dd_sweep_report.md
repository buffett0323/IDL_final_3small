# DD Gate Sweep — Comparison Report

## Experiment Configuration

| Parameter          | Value |
|--------------------|-------|
| dataset            | rand100 (100 random WMT19 EN→ZH sentences) |
| base model         | `NLLB-200-distilled-600M` |
| wait-k             | 5 |
| DD futures K       | 4 (truncation: prefix + 1…4 extra src words) |
| DD steps (n_steps) | 3 — avg JS over first 3 Chinese next-token distributions |
| DD main score      | avg_js_firstN (avg JS over first N=3 steps) |
| future mode        | deterministic truncation from oracle full source |
| taus swept         | 0.01, 0.02, 0.03, 0.05, 0.08, 0.10 |
| gating rule        | COMMIT if avg_js_firstN ≤ τ, else READ |

---

## Aggregate Results

| Run             | BLEU  | AL    | AP    | DAL   | LAAL  |
|-----------------|-------|-------|-------|-------|-------|
| baseline (wait-k=5) | 13.46 |  8.80 | 0.583 |  7.53 |  8.82 |
| dd τ=0.01       | 16.53 | 11.85 | 0.641 | 11.90 | 11.86 |
| dd τ=0.02       | 15.82 | 10.96 | 0.627 | 10.89 | 10.98 |
| dd τ=0.03       | 15.66 | 10.74 | 0.623 | 10.62 | 10.75 |
| dd τ=0.05       | 15.45 | 10.41 | 0.619 | 10.28 | 10.42 |
| dd τ=0.08       | 15.30 | 10.26 | 0.617 | 10.14 | 10.27 |
| dd τ=0.10       | 15.17 | 10.15 | 0.615 |  9.96 | 10.16 |

> **AL / LAAL / DAL**: lower = less latency (better). **BLEU**: higher = better quality.
> **AP** (Average Proportion of source read before each commit): lower = reads less source before committing.

---

### Deltas vs Baseline

| Run        | ΔBLEU  | ΔAL    | ΔLAAL  | BLEU-gain/AL-cost |
|------------|--------|--------|--------|-------------------|
| dd τ=0.01  | +3.07  | +3.05  | +3.04  | 1.01              |
| dd τ=0.02  | +2.36  | +2.16  | +2.16  | 1.09              |
| dd τ=0.03  | +2.19  | +1.94  | +1.93  | 1.13              |
| dd τ=0.05  | +1.98  | +1.61  | +1.61  | 1.23              |
| dd τ=0.08  | +1.84  | +1.46  | +1.45  | 1.26              |
| dd τ=0.10  | +1.71  | +1.35  | +1.34  | 1.27              |

*BLEU-gain/AL-cost = ΔBLEU / ΔAL — higher means more quality per unit of extra latency.*

---

## Best Threshold Analysis

| Criterion                       | Best τ | Value |
|---------------------------------|--------|-------|
| Highest BLEU                    | 0.01   | 16.53 |
| Best BLEU-gain per latency unit | 0.10   | 1.27 BLEU/AL |
| Lowest extra AL vs. baseline    | 0.10   | +1.35 AL |

**Key finding**: The DD gate **consistently improves BLEU at all tested thresholds** (+1.7 to +3.1 BLEU points over baseline). Every improvement comes at a latency cost, but the tradeoff is most efficient at **τ = 0.10** — each extra AL unit buys the most BLEU there.

**Recommendation**: If maximum quality is the goal (latency budget allows), use **τ = 0.01** (BLEU=16.53). For the best quality-latency tradeoff, use **τ = 0.10** (BLEU=15.17, ΔAL=+1.35, efficiency 1.27 BLEU/AL). The baseline wait-k=5 achieves the lowest latency (AL=8.8) but sacrifices ~2 BLEU.

---

## DD Signal Analysis

The gate is evaluated once per unique (sentence, src_len) pair (subsequent calls at the same src_len are cached). Across 100 sentences:

| Run        | Approx. gate calls | Observed behavior |
|------------|-------------------|-------------------|
| dd τ=0.01  | ~1670             | Mostly READs — avg_js_firstN mostly > 0.01 |
| dd τ=0.05  | ~2928             | Mixed COMMIT/READ — threshold sits in the mid-JS range |
| dd τ=0.10  | ~1670             | Mostly COMMITs — only high-JS positions trigger READ |

### What the JS scores look like in practice

Examining the tau=0.05 trace reveals three characteristic patterns:

1. **Near-zero JS (converged futures)** — source prefix is unambiguous, futures all produce the same completion.
   - Example: `"...Ryder Cup defeating Team USA by a final score of 16.5 to 10.5 at Le Golf National outside Paris,"` → avg_js_firstN ≈ 1e-8 → **COMMIT**
   - All 4 futures are identical at this point; the model is certain.

2. **Low JS (< 0.05)** — prefix is fairly stable; slight context extension changes little.
   - Example: `"AMs are apparently suggesting alternative options,"` → avg_js_first1=0.004, avg_js_firstN=0.037 → **COMMIT** (at τ=0.05)
   - avg_js_first1 is very small, but avg_js_firstN (3-step horizon) is larger because the 3rd token diverges more.

3. **High JS (> 0.05)** — prefix is ambiguous; extra source words substantially change next-token predictions.
   - Example: `"Researchers...in order to win government"` → avg_js_firstN=0.25 → **READ** (τ=0.05)
   - Futures diverge: "grants," vs "Lee said." produce very different Chinese distributions.
   - Example: `"Patches can be found about 10 miles offshore"` → avg_js_firstN=0.12 → **READ** (τ=0.05)
   - "offshore of Hillsborough County" is not predictable from "offshore" alone.

### avg_js_first1 vs avg_js_firstN divergence

A consistent pattern: **avg_js_first1 is usually much smaller than avg_js_firstN**. Example:

| Position | avg_js_first1 | avg_js_firstN |
|----------|--------------|--------------|
| "AMs ... options," | 0.004 | 0.037 |
| "AMs ... but the" | 0.002 | 0.033 |
| "Patches ... offshore" | 0.006 | 0.115 |

This shows that the **first next token is often stable** (the model commits confidently to a first Chinese character), but **further tokens in the sequence diverge** depending on what source context follows. Using `avg_js_firstN` (N=3) therefore gives a stronger, more discriminative signal than `avg_js_first1` alone.

---

## Verbose Example Gate Decisions (τ = 0.05)

### 5 cases where DD forced READ (high divergence — ambiguous prefix)

---

**sent=31  src_len=13  tgt_len=8**
```
Observed prefix : "Researchers in the U.S. often have to work hard in order to win"
future[1]       : "Researchers in the U.S. often have to work hard in order to win government"
future[2]       : "Researchers in the U.S. often have to work hard in order to win government grants,"
future[3]       : "Researchers in the U.S. often have to work hard in order to win government grants, Lee"
future[4]       : "Researchers in the U.S. often have to work hard in order to win government grants, Lee said."
avg_js_first1=0.0497  avg_js_first3=0.2018  avg_js_firstN=0.2018
per_step_js   : [0.050, 0.241, 0.315]
DD threshold  : 0.05  →  READ  (baseline would: COMMIT)
```
*Why*: Knowing just "win" the model doesn't know if next is "government," "an award," or other; more context drastically changes the Chinese output for "government grants."

---

**sent=31  src_len=14  tgt_len=8**
```
Observed prefix : "Researchers in the U.S. often have to work hard in order to win government"
future[1]       : "Researchers in the U.S. often have to work hard in order to win government grants,"
future[2]       : "Researchers in the U.S. often have to work hard in order to win government grants, Lee"
future[3]       : "Researchers in the U.S. often have to work hard in order to win government grants, Lee said."
future[4]       : "Researchers in the U.S. often have to work hard in order to win government grants, Lee said."
avg_js_first1=0.0475  avg_js_first3=0.2517  avg_js_firstN=0.2517
per_step_js   : [0.047, 0.293, 0.415]
DD threshold  : 0.05  →  READ  (baseline would: COMMIT)
```
*Why*: Even knowing "government", "grants" vs the end-of-sentence is critical for Chinese word choice. Futures still diverge at step 3.

---

**sent=100  src_len=5  tgt_len=0**
```
Observed prefix : "Patches can be found about"
future[1]       : "Patches can be found about 10"
future[2]       : "Patches can be found about 10 miles"
future[3]       : "Patches can be found about 10 miles offshore"
future[4]       : "Patches can be found about 10 miles offshore of"
avg_js_first1=0.0153  avg_js_first3=0.1206  avg_js_firstN=0.1206
per_step_js   : [0.015, 0.027, 0.319]
DD threshold  : 0.05  →  READ  (baseline would: COMMIT)
```
*Why*: "Patches can be found about" is an incomplete prepositional phrase; step-3 futures diverge enormously (0.319 JS) because without knowing "10 miles offshore", the Chinese translation of location/measurement can't be determined.

---

**sent=100  src_len=11  tgt_len=1**
```
Observed prefix : "Patches can be found about 10 miles offshore of Hillsborough County,"
future[1]       : "Patches can be found about 10 miles offshore of Hillsborough County, but"
future[2]       : "Patches can be found about 10 miles offshore of Hillsborough County, but at"
future[3]       : "Patches can be found about 10 miles offshore of Hillsborough County, but at fewer"
future[4]       : "Patches can be found about 10 miles offshore of Hillsborough County, but at fewer sites"
avg_js_first1=0.0019  avg_js_first3=0.1101  avg_js_firstN=0.1101
per_step_js   : [0.002, 0.003, 0.325]
DD threshold  : 0.05  →  READ  (baseline would: COMMIT)
```
*Why*: avg_js_first1 is tiny (0.002) but avg_js_firstN=0.11 because the 3rd token in the Chinese output varies with context. The phrase "but at fewer sites" is hard to predict without reading more.

---

**sent=1  src_len=5  tgt_len=0**
```
Observed prefix : "AMs are apparently suggesting alternative"
future[1]       : "AMs are apparently suggesting alternative options,"
future[2]       : "AMs are apparently suggesting alternative options, but"
future[3]       : "AMs are apparently suggesting alternative options, but the"
future[4]       : "AMs are apparently suggesting alternative options, but the struggle"
avg_js_first1=0.0099  avg_js_first3=0.0431  avg_js_firstN=0.0431
per_step_js   : [0.010, 0.021, 0.098]
DD threshold  : 0.05  →  COMMIT  (same at τ=0.05, but READ at τ=0.01)
```
*Note*: This is near the threshold boundary; at τ=0.01 this is a READ, at τ=0.05 a COMMIT. Shows threshold sensitivity around JS≈0.04.

---

### 5 cases where DD allowed COMMIT (low divergence — stable prefix)

---

**sent=99  src_len=25  tgt_len=18**
```
Observed prefix : "Team Europe has won the 2018 Ryder Cup...at Le Golf National outside Paris,"
All 4 futures   : [same sentence — EOS reached]
avg_js_firstN=1.7e-08  avg_js_first1=1.7e-08
→  COMMIT  (both DD and baseline agree — sentence complete)
```

**sent=100  src_len=17  tgt_len=1**
```
Observed prefix : "Patches can be found about 10 miles offshore of Hillsborough County, but at fewer sites relative to"
future[1..3]    : "...relative to last week." (converging)
avg_js_firstN=0.0032  avg_js_first1=0.0066
→  COMMIT  (futures agree on "last week." ending)
```

**sent=31  src_len=16  tgt_len=8**
```
Observed prefix : "Researchers in the U.S....to win government grants, Lee"
All 4 futures   : "...Lee said." (EOS)
avg_js_firstN≈0  avg_js_first1=0.0
→  COMMIT  (complete sentence, no ambiguity)
```

**sent=1  src_len=6  tgt_len=1**
```
Observed prefix : "AMs are apparently suggesting alternative options,"
avg_js_firstN=0.0366  avg_js_first1=0.0042
→  COMMIT  (JS below 0.05 threshold; step-1 is very stable)
```

**sent=1  src_len=7  tgt_len=2**
```
Observed prefix : "AMs are apparently suggesting alternative options, but"
avg_js_firstN=0.0426  avg_js_first1=0.0032
→  COMMIT  (consistently below τ=0.05 despite higher 3-step JS)
```

---

## Observations and Failure Mode Analysis

### 1. DD Gate Successfully Discriminates Ambiguous vs Stable Positions

The gate works as designed: high JS coincides with genuinely ambiguous prefixes where future context changes the translation. This is validated by the examples above (e.g., "win government" vs "win government grants", "Patches can be found about" before knowing the location).

### 2. avg_js_first3/firstN is a Stronger Signal than avg_js_first1

Across all examples, avg_js_first1 tends to be small (≈0.001–0.015) while avg_js_firstN is much larger (0.03–0.43). The first Chinese token after any prefix is often relatively stable; it's the **continuation** (tokens 2 and 3) that diverges. Consequently:
- Using only avg_js_first1 would cause many missed READs (false commits at ambiguous positions)
- Using avg_js_firstN (3-step) gives a more reliable uncertainty signal

### 3. Threshold Sensitivity and Operating Point

| τ     | Behavior |
|-------|----------|
| 0.01  | Almost always READ (JS > 0.01 at almost every step) → nearly reads full sentence before committing → highest BLEU, highest latency |
| 0.05  | Fires selectively — triggers at genuine high-uncertainty prefixes (e.g., incomplete noun phrases, proper noun contexts) |
| 0.10  | Fires rarely — only at very ambiguous positions → more commits → moderate latency savings |

### 4. Root Cause of Quality-Latency Tradeoff

The baseline wait-k=5 commits aggressively after every 5th source word. Many of those commits happen at genuinely uncertain positions (e.g., "AMs are apparently suggesting alternative" is an incomplete phrase). The DD gate delays those commits, reading 1–5 more words until the future distributions stabilize. This produces better translations at the cost of those extra reads.

### 5. Expected Failure Modes

- **JS always near-zero**: Would occur if futures always converge immediately (e.g., very short sentences or highly predictable corpora). DD would add no benefit over baseline.
- **JS always high**: Would occur if the model is always uncertain regardless of prefix length (e.g., very ambiguous source language). DD would force full source read before any commit → effectively an offline translation system.
- **avg_js_first1 vs avg_js_firstN mismatch**: If the task required only 1-step lookahead (fast response), first1 might be sufficient but would miss many relevant READs. For 3+ token lookahead tasks, firstN is critical.

### 6. Next Steps

1. **Run tau=0.04, 0.06, 0.07, 0.09** to fill gaps in the sweep curve — currently missing these 4 data points.
2. **Try tau=0.15 or tau=0.20** to see if BLEU-gain/AL-cost ratio continues to improve beyond τ=0.10.
3. **Ablate K**: compare K=2 vs K=4 vs K=8 futures — K=4 already shows strong signal; K=2 may be sufficient with similar quality at half the compute cost.
4. **Combine DD with entropy gate**: use `DD AND existing_entropy → more selective READ` to reduce unnecessary latency while keeping quality gains.
5. **LM-sampled futures**: Replace deterministic truncation futures with stochastic completions from a small LM for more genuine diversity in the JS estimate.
6. **Evaluate on COMET**: The BLEU improvements (+1.7–+3.1) may be even more pronounced in COMET due to BLEU's n-gram limitations with Chinese.

---

## Summary

The DD gate ablation is a **confirmed success** in the oracle source setting:

- **BLEU improves at every threshold tested** (+1.71 to +3.07 over baseline)
- The improvement is **monotonic** with decreasing τ: lower τ = more reads = higher quality
- The **best tradeoff (BLEU-gain per AL-cost unit)** is at τ=0.10, giving +1.71 BLEU at +1.35 AL cost (efficiency=1.27 BLEU/AL)
- The DD signal is **semantically meaningful**: high JS correctly identifies genuinely ambiguous prefixes where reading more improves translation
- The **avg_js_firstN (3-step)** signal is substantially more discriminative than avg_js_first1 alone

**Main caveat**: This uses oracle source (full sentence known at test time for computing DD), which is an upper bound. Online DD would require an English LM to sample futures instead of truncation from the oracle. The oracle experiment confirms the signal is valid; practical deployment requires replacing oracle truncation with sampled futures.
