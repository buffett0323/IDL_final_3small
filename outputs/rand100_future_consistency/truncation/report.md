# Future-Consistency Experiment Report

> **⚠ Future mode: TRUNCATION** — futures are nested prefix extensions of
> the observed prefix.  This measures *same-path prefix-extension stability*,
> **not** branching future robustness.  Do not interpret COMMIT as robust
> to genuinely different source continuations.

## 1. Experiment Configuration

| Parameter           | Value |
|---------------------|-------|
| base_model          | `/data/user_data/haolingp/models/Qwen3-4B-Base` |
| causal_lm           | `True` |
| device              | `cuda:0` |
| future_strategy     | truncation (reveal 1..K more source words) |
| K (num_futures)     | `4` |
| prefix_words        | `5` |
| cont_len            | `8` |
| top_k_overlap       | `10` |
| dataset             | rand100 (100 random WMT19 EN→ZH sentences) |
| n_examples          | `50` |
| commit_js_threshold | `0.05` |
| read_js_threshold   | `0.2` |
| commit_lcp_min      | `2` |
| read_edit_threshold | `0.8` |
| future_mode         | `truncation` |

## 1b. Future Diversity Diagnostics

| Metric                  | Mean   | Note |
|-------------------------|--------|------|
| nested_prefix_rate      | 0.883  | 1.0 = all pairs nested (truncation always gives 1.0) |
| avg_future_edit_dist    | 0.173  | 0 = identical, 1 = maximally different |

**All 50 examples use nested truncation futures.**
Results measure *same-path continuation stability only*, NOT branching robustness.

## 2. Distribution Divergence Results

| Metric            | Mean   | Median | Std    |
|-------------------|--------|--------|--------|
| avg_js_divergence | 0.0121 | 0.0050 | 0.0160 |
| max_js_divergence | 0.0231 | 0.0092 | 0.0283 |
| topk_overlap      | 0.8480 | 0.9000 | 0.1129 |

**Decisions**: COMMIT=47 (94%)  BORDERLINE=3 (6%)  READ=0 (0%)

## 3. Semantic LCP Results

| Metric              | Mean   | Median | Std    |
|---------------------|--------|--------|--------|
| literal_lcp_len     | 4.96 | 5.00 | 3.68 |
| avg_edit_distance   | 0.2023 | 0.1491 | 0.2136 |
| semantic_agreement  | 0.7355 | 0.7350 | 0.2686 |
| non-empty safe prefix | 41/50 (82%) | — | — |

**Decisions**: COMMIT=35 (70%)  BORDERLINE=15 (30%)  READ=0 (0%)

## 4. Qualitative Examples

### 4a. Strong Agreement (low JS divergence → stable futures)

- **Example 20** (JS=0.0000, LCP=8)  
  `Rangers were short of inspiration, though.`  
  LCP=`瑞恩士队虽然缺乏` | dd=COMMIT | lcp=COMMIT

- **Example 24** (JS=0.0000, LCP=7)  
  `That has not happened yet.`  
  LCP=`这还没有发生。` | dd=COMMIT | lcp=COMMIT

- **Example 60** (JS=0.0000, LCP=9)  
  `Germany's new warship postponed yet again`  
  LCP=`德国新战舰再次推迟` | dd=COMMIT | lcp=COMMIT

- **Example 90** (JS=0.0000, LCP=9)  
  `He died later in hospital.`  
  LCP=`他后来在医院去世。` | dd=COMMIT | lcp=COMMIT

- **Example 96** (JS=0.0000, LCP=9)  
  `Pyongyang has made no public comments.`  
  LCP=`北京没有公开评论。` | dd=COMMIT | lcp=COMMIT

### 4b. Strong Disagreement (high JS divergence → risky futures)

- **Example 52** (JS=0.0724, LCP=0)  
  `The perennially topical premise now is that Great Britain is in serious trouble.`  
  dd=BORDERLINE | lcp=BORDERLINE

- **Example 62** (JS=0.0568, LCP=2)  
  `Expert sound analysis of all recordings will ascertain the frequency of the bat `  
  dd=BORDERLINE | lcp=BORDERLINE

- **Example 34** (JS=0.0533, LCP=0)  
  `On the importance of a responsible economic policy, on national security, on Eur`  
  dd=BORDERLINE | lcp=BORDERLINE

- **Example 18** (JS=0.0360, LCP=0)  
  `Voters chose overwhelmingly to become independent, though turnout was low with t`  
  dd=COMMIT | lcp=BORDERLINE

- **Example 40** (JS=0.0347, LCP=0)  
  `"It's not a good sign for Morrisey that the president has to come to try to give`  
  dd=COMMIT | lcp=BORDERLINE

### 4c. Borderline Examples

- **Example 34** (JS=0.0533, overlap=0.70, LCP=0)  
  `On the importance of a responsible economic policy, on national security, on Eur`  
  dd=BORDERLINE | lcp=BORDERLINE

- **Example 52** (JS=0.0724, overlap=0.70, LCP=0)  
  `The perennially topical premise now is that Great Britain is in serious trouble.`  
  dd=BORDERLINE | lcp=BORDERLINE

- **Example 62** (JS=0.0568, overlap=0.80, LCP=2)  
  `Expert sound analysis of all recordings will ascertain the frequency of the bat `  
  dd=BORDERLINE | lcp=BORDERLINE

## 5. Observations

### Patterns

- Distribution divergence: 0.0121 mean JS. Low — futures are broadly consistent.
- LCP: 5.0 mean chars in common. Non-trivial agreement on the near-term Chinese continuation.
- Top-k overlap: 84.80% of top-10 tokens shared — good token agreement.

### Likely Failure Modes

- Truncation futures that differ only in trailing punctuation/articles may
  produce artificially low JS divergence; true ambiguity cases (pronoun
  resolution, named entity completion) would need longer continuations.
- LCP is strictly literal: semantically equivalent paraphrases (e.g. 病院 vs
  医院) score zero agreement even when meaning is identical.
- Qwen4B few-shot prompt may repeat the next few-shot line for very short
  inputs; continuation cleaning strips this but may lose valid content.

### Next Steps

1. Compare future-consistency signals to per-step entropy on the same examples.
2. Use future-consistency as the gating signal instead of entropy threshold.
3. Try LM-sampled English futures (top-p sampling) instead of truncation.
4. Add embedding-based semantic clustering for a stronger 'semantic LCP'.
