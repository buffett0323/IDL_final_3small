# Wait-k vs DD Full Gate vs DD Veto — Comparison Report

## Experiment Setup

| Parameter       | Value                           |
|-----------------|----------------------------------|
| Dataset         | rand100 (100 WMT19 EN→ZH)        |
| Base model      | NLLB-200-distilled-600M          |
| DD futures K    | 4 (truncation mode)              |
| DD steps        | 3 (avg_js_first3 = policy score) |
| Baseline method | wait-k, entropy gate τ=999 (off) |
| Veto baseline   | wait-k + entropy gate τ=3.0      |

### Methods

- **baseline wait-k**: pure wait-k policy, no uncertainty gate, varying k.
- **DD full gate**: post-wait-k, DD is the sole commit/read arbiter.
  DD score (avg JS divergence) > τ → READ; else → COMMIT.
- **DD veto**: baseline (wait-k + entropy gate) decides first. Only when
  baseline would COMMIT does DD get to veto if divergence is high.

## Results

| Method             | BLEU  | AL    | LAAL  | AP    | DAL   | Read% |
|--------------------|-------|-------|-------|-------|-------|-------|
| baseline k=3       | 12.72 | 7.34 | 7.36 | 0.541 | 5.99 |     — |
| baseline k=5       | 13.46 | 8.80 | 8.82 | 0.583 | 7.53 |     — |
| baseline k=7       | 14.30 | 10.24 | 10.25 | 0.616 | 9.04 |     — |
| baseline k=9       | 14.69 | 11.60 | 11.60 | 0.642 | 10.49 |     — |
| DD full τ=0.03     | 15.66 | 10.74 | 10.75 | 0.623 | 10.62 |   26% |
| DD full τ=0.05     | 15.45 | 10.41 | 10.42 | 0.619 | 10.28 |   23% |
| entropy only       | 14.92 | 10.35 | 10.36 | 0.632 | 9.35 |     — |
| DD veto τ=0.03     | 16.94 | 11.96 | 11.97 | 0.660 | 12.00 |   24% |
| DD veto τ=0.05     | 16.73 | 11.65 | 11.66 | 0.656 | 11.68 |   22% |
| NLLB continuation k=5 | 8.67 | 8.68 | 8.71 | 0.531 | 7.03 |     — |
| Qwen continuation k=5 | 8.46 | 8.70 | 9.04 | 0.784 | 9.65 |     — |

## DD Gate Signal

| Method             | Gate calls | READ%  | Avg JS (firstN) |
|--------------------|------------|--------|-----------------|
| DD full τ=0.03     |       1669 |  25.5% |          0.0418 |
| DD full τ=0.05     |       1669 |  23.5% |          0.0418 |
| DD veto τ=0.03     |       1415 |  24.5% |          0.0400 |
| DD veto τ=0.05     |       1415 |  22.4% |          0.0400 |

## Analysis

### Baseline sweep
- Best BLEU : **DD veto τ=0.05** BLEU=14.69 AL=11.60
- Lowest AL : **baseline k=3** AL=7.34 BLEU=12.72

Reference (baseline k=5): BLEU=13.46, AL=8.80

### DD full gate
- **DD full τ=0.03**: BLEU ↑+2.19 (meaningful gain), AL +1.94 (higher latency)
- **DD full τ=0.05**: BLEU ↑+1.99 (meaningful gain), AL +1.61 (higher latency)

### DD veto
- **DD veto τ=0.03**: BLEU ↑+3.47 (meaningful gain), AL +3.16 (higher latency)
- **DD veto τ=0.05**: BLEU ↑+3.27 (meaningful gain), AL +2.85 (higher latency)

## Plot

See `outputs/comparison_plot.png` for the quality-latency scatter plot.
X-axis: Average Lagging (AL) — lower is better (less latency).
Y-axis: BLEU score — higher is better.
Each method is shown as a curve through its operating points.
