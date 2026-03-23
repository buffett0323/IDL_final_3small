# Future-Consistency: Truncation vs LM-Sample Comparison

This report explicitly distinguishes:
- **Same-path stability** (truncation futures, nested prefix extensions)
- **Branching robustness** (lm_sample futures, genuinely diverse continuations)

Do NOT claim branching robustness based on truncation results.

n=50 examples, K=4, prefix=5 words, cont_len=8

## 1. Future Diversity

| Metric                    | truncation | lm_sample |
|---------------------------|------------|-----------|
| nested_prefix_rate (mean) | 0.883       | 0.000      |
| avg_future_edit_dist      | 0.173       | 0.531      |
| unique_future_ratio       | 0.905       | 1.000      |

lm_sample fell back to nested futures in 0/50 examples (0%).

## 2. Distribution Divergence (JS)

| Metric             | truncation | lm_sample | Δ (lm−trunc) |
|--------------------|------------|-----------|--------------|
| avg_js  mean       | 0.0121     | 0.0415    | +0.0294       |
| avg_js  median     | 0.0050     | 0.0238    | +0.0188       |
| topk_overlap mean  | 0.8480     | 0.7460    | -0.1020       |

- DD COMMIT: truncation=47 (94%)  lm_sample=38 (76%)
- DD BORDERLINE: truncation=3 (6%)  lm_sample=11 (22%)
- DD READ: truncation=0 (0%)  lm_sample=1 (2%)

✓ lm_sample futures produce **higher JS divergence** — diverse futures
  expose more genuine uncertainty than truncation futures.

## 3. Semantic LCP

| Metric                | truncation | lm_sample | Δ |
|-----------------------|------------|-----------|---|
| literal_lcp_len mean  | 4.96       | 3.44      | -1.52 |
| avg_edit_dist mean    | 0.2023     | 0.4692    | +0.2670 |
| semantic_agreement    | 0.7355     | 0.4808    | -0.2547 |

- LCP COMMIT: truncation=35 (70%)  lm_sample=22 (44%)
- LCP BORDERLINE: truncation=15 (30%)  lm_sample=24 (48%)
- LCP READ: truncation=0 (0%)  lm_sample=4 (8%)

## 4. Decision Changes (truncation→lm_sample)

12/50 examples changed DD decision when switching to lm_sample futures:

- Example 10: COMMIT → BORDERLINE  | `A farmer was attacked and killed by a pig in a market in sou`
- Example 16: COMMIT → BORDERLINE  | `Ford told the Senate Judiciary Committee that she's 100 perc`
- Example 18: COMMIT → BORDERLINE  | `Voters chose overwhelmingly to become independent, though tu`
- Example 34: BORDERLINE → READ  | `On the importance of a responsible economic policy, on natio`
- Example 52: BORDERLINE → COMMIT  | `The perennially topical premise now is that Great Britain is`
- Example 54: COMMIT → BORDERLINE  | `District of Columbia Attorney General Karl Racine said in a `
- Example 58: COMMIT → BORDERLINE  | `"When you go through so much at such a young age" - he went `
- Example 80: COMMIT → BORDERLINE  | `Vice President Mike Pence is now slated to address the confe`
- Example 84: COMMIT → BORDERLINE  | `"We think the decision to seek the death penalty rather than`
- Example 86: COMMIT → BORDERLINE  | `The premier is struggling to keep her Chequers compromise pl`
- ... and 2 more (see results JSONL files)

## 5. Interpretation

| Claim | Supported by | Valid? |
|-------|-------------|--------|
| 'Same-path stable' (truncation COMMIT=94%) | truncation mode | ✓ Valid |
| 'Branching robust' (lm_sample COMMIT=76%) | lm_sample mode | ✓ Valid |

**Key takeaway**: Use truncation results to justify same-path stability only.
Use lm_sample results to justify branching future robustness.
