# Future-Consistency Experiment Report

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
| n_examples          | `3` |
| commit_js_threshold | `0.05` |
| read_js_threshold   | `0.2` |
| commit_lcp_min      | `2` |
| read_edit_threshold | `0.8` |

## 2. Distribution Divergence Results

| Metric            | Mean   | Median | Std    |
|-------------------|--------|--------|--------|
| avg_js_divergence | 0.0064 | 0.0051 | 0.0037 |
| max_js_divergence | 0.0129 | 0.0098 | 0.0076 |
| topk_overlap      | 0.8667 | 0.9000 | 0.0577 |

**Decisions**: COMMIT=3 (100%)  BORDERLINE=0 (0%)  READ=0 (0%)

## 3. Semantic LCP Results

| Metric              | Mean   | Median | Std    |
|---------------------|--------|--------|--------|
| literal_lcp_len     | 4.33 | 0.00 | 7.51 |
| avg_edit_distance   | 0.1462 | 0.1250 | 0.1579 |
| semantic_agreement  | 0.7371 | 0.5694 | 0.4138 |
| non-empty safe prefix | 1/3 (33%) | — | — |

**Decisions**: COMMIT=1 (33%)  BORDERLINE=2 (67%)  READ=0 (0%)

## 4. Qualitative Examples

### 4a. Strong Agreement (low JS divergence → stable futures)

- **Example 0** (JS=0.0036, LCP=13)  
  `AMs are apparently suggesting alternative options, but the struggle to reach con`  
  LCP=`AMs似乎在提出替代方案，` | dd=COMMIT | lcp=COMMIT

- **Example 2** (JS=0.0051, LCP=0)  
  `Giles added the victim sustained traumatic injuries to his upper torso area.`  
  LCP=`` | dd=COMMIT | lcp=BORDERLINE

- **Example 1** (JS=0.0105, LCP=0)  
  `The boy was airlifted to Rady Children's Hospital in San Diego where he is liste`  
  LCP=`` | dd=COMMIT | lcp=BORDERLINE

### 4b. Strong Disagreement (high JS divergence → risky futures)

- **Example 1** (JS=0.0105, LCP=0)  
  `The boy was airlifted to Rady Children's Hospital in San Diego where he is liste`  
  dd=COMMIT | lcp=BORDERLINE

- **Example 2** (JS=0.0051, LCP=0)  
  `Giles added the victim sustained traumatic injuries to his upper torso area.`  
  dd=COMMIT | lcp=BORDERLINE

- **Example 0** (JS=0.0036, LCP=13)  
  `AMs are apparently suggesting alternative options, but the struggle to reach con`  
  dd=COMMIT | lcp=COMMIT

### 4c. Borderline Examples

- **Example 0** (JS=0.0036, overlap=0.80, LCP=13)  
  `AMs are apparently suggesting alternative options, but the struggle to reach con`  
  dd=COMMIT | lcp=COMMIT

- **Example 1** (JS=0.0105, overlap=0.90, LCP=0)  
  `The boy was airlifted to Rady Children's Hospital in San Diego where he is liste`  
  dd=COMMIT | lcp=BORDERLINE

- **Example 2** (JS=0.0051, overlap=0.90, LCP=0)  
  `Giles added the victim sustained traumatic injuries to his upper torso area.`  
  dd=COMMIT | lcp=BORDERLINE

## 5. Observations

### Patterns

- Distribution divergence: 0.0064 mean JS. Low — futures are broadly consistent.
- LCP: 4.3 mean chars in common. Non-trivial agreement on the near-term Chinese continuation.
- Top-k overlap: 86.67% of top-10 tokens shared — good token agreement.

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
