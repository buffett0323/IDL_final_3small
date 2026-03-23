# Future-Consistency Experiment Report

> **Future mode: LM_SAMPLE** — futures are sampled from the base LM with
> temperature sampling and nested-prefix rejection.  Results measure
> *branching future robustness*.

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
| future_mode         | `lm_sample` |
| lm_temperature      | `1.2` |
| lm_top_p            | `0.9` |

## 1b. Future Diversity Diagnostics

| Metric                  | Mean   | Note |
|-------------------------|--------|------|
| nested_prefix_rate      | 0.000  | 1.0 = all pairs nested (truncation always gives 1.0) |
| avg_future_edit_dist    | 0.531  | 0 = identical, 1 = maximally different |

0/50 examples fell back to nested futures.
50/50 examples have genuinely diverse branching futures.

## 2. Distribution Divergence Results

| Metric            | Mean   | Median | Std    |
|-------------------|--------|--------|--------|
| avg_js_divergence | 0.0415 | 0.0238 | 0.0496 |
| max_js_divergence | 0.0769 | 0.0415 | 0.0931 |
| topk_overlap      | 0.7460 | 0.8000 | 0.1460 |

**Decisions**: COMMIT=38 (76%)  BORDERLINE=11 (22%)  READ=1 (2%)

## 3. Semantic LCP Results

| Metric              | Mean   | Median | Std    |
|---------------------|--------|--------|--------|
| literal_lcp_len     | 3.44 | 2.50 | 3.21 |
| avg_edit_distance   | 0.4692 | 0.4451 | 0.2589 |
| semantic_agreement  | 0.4808 | 0.4393 | 0.2882 |
| non-empty safe prefix | 35/50 (70%) | — | — |

**Decisions**: COMMIT=22 (44%)  BORDERLINE=24 (48%)  READ=4 (8%)

## 4. Qualitative Examples

### 4a. Strong Agreement (low JS divergence → stable futures)

- **Example 60** (JS=0.0005, LCP=2)  
  `Germany's new warship postponed yet again`  
  LCP=`德国` | dd=COMMIT | lcp=BORDERLINE

- **Example 36** (JS=0.0010, LCP=5)  
  `But Melrose began the second half well and Patrick Anderson's try, converted by `  
  LCP=`但梅洛斯在` | dd=COMMIT | lcp=COMMIT

- **Example 74** (JS=0.0014, LCP=6)  
  `Another commented: "This is a fun advert aimed at mums who pump (often in their `  
  LCP=`另一个人评论` | dd=COMMIT | lcp=COMMIT

- **Example 22** (JS=0.0017, LCP=3)  
  `Trump argued that Democrats are on a mission to "resist and obstruct."`  
  LCP=`特朗普` | dd=COMMIT | lcp=COMMIT

- **Example 66** (JS=0.0025, LCP=5)  
  `Trump said he would "prefer not" to fire Rosenstein but then the meeting was del`  
  LCP=`特朗普表示` | dd=COMMIT | lcp=COMMIT

### 4b. Strong Disagreement (high JS divergence → risky futures)

- **Example 34** (JS=0.2003, LCP=0)  
  `On the importance of a responsible economic policy, on national security, on Eur`  
  dd=READ | lcp=READ

- **Example 96** (JS=0.1545, LCP=2)  
  `Pyongyang has made no public comments.`  
  dd=BORDERLINE | lcp=BORDERLINE

- **Example 80** (JS=0.1478, LCP=7)  
  `Vice President Mike Pence is now slated to address the conference, now scheduled`  
  dd=BORDERLINE | lcp=COMMIT

- **Example 62** (JS=0.1404, LCP=0)  
  `Expert sound analysis of all recordings will ascertain the frequency of the bat `  
  dd=BORDERLINE | lcp=READ

- **Example 16** (JS=0.1302, LCP=2)  
  `Ford told the Senate Judiciary Committee that she's 100 percent certain that Kav`  
  dd=BORDERLINE | lcp=BORDERLINE

### 4c. Borderline Examples

- **Example 10** (JS=0.1296, overlap=0.50, LCP=1)  
  `A farmer was attacked and killed by a pig in a market in southwest China, accord`  
  dd=BORDERLINE | lcp=BORDERLINE

- **Example 16** (JS=0.1302, overlap=0.70, LCP=2)  
  `Ford told the Senate Judiciary Committee that she's 100 percent certain that Kav`  
  dd=BORDERLINE | lcp=BORDERLINE

- **Example 18** (JS=0.0524, overlap=0.50, LCP=3)  
  `Voters chose overwhelmingly to become independent, though turnout was low with t`  
  dd=BORDERLINE | lcp=BORDERLINE

- **Example 54** (JS=0.0875, overlap=0.60, LCP=8)  
  `District of Columbia Attorney General Karl Racine said in a statement Friday tha`  
  dd=BORDERLINE | lcp=COMMIT

- **Example 58** (JS=0.1121, overlap=0.60, LCP=0)  
  `"When you go through so much at such a young age" - he went to his first Olympic`  
  dd=BORDERLINE | lcp=BORDERLINE

## 5. Observations

### Patterns

- Distribution divergence: 0.0415 mean JS. Low — futures are broadly consistent.
- LCP: 3.4 mean chars in common. Non-trivial agreement on the near-term Chinese continuation.
- Top-k overlap: 74.60% of top-10 tokens shared — good token agreement.

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
