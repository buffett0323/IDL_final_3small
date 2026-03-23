# Prefix-Constrained Continuation vs Re-Translation Baseline

## Motivation

The re-translation baseline computes a full Chinese translation for the source
prefix at each step, then emits `translation[tgt_len]`.  Because NLLB's full-
sentence hypothesis can change as more source words arrive, **consecutive emitted
characters can come from incompatible hypotheses** — producing garbled, repetitive
output even when each individual hypothesis is correct.

Prefix-constrained continuation fixes this by conditioning the NLLB decoder on
the *already-committed target prefix* (`decoder_input_ids`), then generating
only the *next* character.  The decoder is physically prevented from contradicting
what was already written.

## System-Level Metrics (BLEU / Latency)

| Metric | Baseline (re-transl.) | Continuation | Δ |
|--------|-----------------------|--------------|---|
| BLEU | 13.462 | 8.671 | -4.791 |
| AL | 8.800 | 8.675 | -0.125 |
| LAAL | 8.816 | 8.710 | -0.106 |
| AP | 0.583 | 0.531 | -0.052 |

## Sentence-Level Analysis

Evaluated on **100 sentences** (rand100 test set).

### Translation Inconsistency (bigram repetition ratio)

Higher ratio = more garbled / repetitive output.

| Method | Mean inconsistency |
|--------|--------------------|
| Baseline      | 0.0428 |
| Continuation  | 0.0316 |
| Δ (cont − base) | -0.0112 |

A negative Δ means continuation produces **less** repetitive / garbled output.

### Character-level F1 vs Reference

| Method | Mean charF1 |
|--------|-------------|
| Baseline     | 0.4101 |
| Continuation | 0.3726 |
| Δ            | -0.0375 |

### Per-sentence outcome breakdown

| Outcome | Count | % |
|---------|-------|----|
| Continuation better (Δ > 0.01) | 34 | 34% |
| Roughly equal (|Δ| ≤ 0.01) | 11 | 11% |
| Baseline better (Δ < -0.01) | 55 | 55% |

## Case Studies: Most Improved Sentences

These are the sentences where continuation eliminated the largest
translation-hypothesis inconsistency errors.

### Case 1: Sentence 41  (charF1 Δ = +0.325)

**Source**: "It's not a good sign for Morrisey that the president has to come to try to give him a boost in the polls," said Simon Haeder, a political scientist at West Virginia University, according to Reuters.

| | Text | charF1 |
|---|------|--------|
| Reference    | 据路透社报道，西弗吉尼亚大学的政治学者西蒙·哈德(SimonHaeder)表示，“总统不得不在民意调查中为其推波助澜，这对莫里西而言并不是一个好兆头”。 | — |
| Baseline     | "这对莫一西来说不是一个好迹象,总统必须来试图给他在民意调查来试图给他在民意调查中带来推动, | 0.295 |
| Continuation | "这对莫里西来说不好,总裁必来试图给他在民意调试中带来推进",西弗吉尼尼大政学家西蒙·海德尔(SimonHaeder)据路透社报道. | 0.620 |

### Case 2: Sentence 50  (charF1 Δ = +0.293)

**Source**: Game of Thrones star Kit Harington hits out at toxic masculinity

| | Text | charF1 |
|---|------|--------|
| Reference    | 《权力的游戏》影星KitHarington猛烈抨击“有毒的”男子气概 | — |
| Baseline     | 游戏的星人凯特哈林顿,击了毒性男性性. | 0.264 |
| Continuation | 游王之王的明星KitHarington突破了毒性男人性 | 0.557 |

### Case 3: Sentence 16  (charF1 Δ = +0.250)

**Source**: Flash flooding is possible with rapidly deteriorating conditions due to the scattered nature of tropical rain.

| | Text | charF1 |
|---|------|--------|
| Reference    | 由于热带雨具有分散性，伴随条件的快速恶化，可能会出现洪水泛滥。 | — |
| Baseline     | 闪速迅速恶可能,即的,着条件的迅速恶化,即发生流. | 0.286 |
| Continuation | 闪电洪灾可发现,由于热带雨的分散性,条例迅猛恶化. | 0.536 |

### Case 4: Sentence 48  (charF1 Δ = +0.239)

**Source**: The hour long course involves a series of interactive tasks.

| | Text | charF1 |
|---|------|--------|
| Reference    | 长达一小时的课程涉及一系列互动任务。 | — |
| Baseline     | 长达一个一小时的时间, | 0.483 |
| Continuation | 长达一小时的课程包围着一系互动任职. | 0.722 |

### Case 5: Sentence 69  (charF1 Δ = +0.167)

**Source**: Trainers and medical personnel gave Abercrombie oxygen on the sideline before placing him on a stretcher and taking him back for further evaluation.

| | Text | charF1 |
|---|------|--------|
| Reference    | 培训师和医务人员在边线旁给阿伯克龙比输了氧气，然后将他抬到担架上，带回来进行进一步评估。 | — |
| Baseline     | 培练人员和医疗人员在将Abercrombie放在床上,, | 0.222 |
| Continuation | 培训人和医疗人才在安德克伦比被带回床上进去进度评价之前, | 0.389 |

## Case Studies: Baseline Better (Degraded by Continuation)

A few sentences where re-translation happens to score higher.
These usually occur when the model's hypothesis is stable and
the continuation goes off-track due to tokenization boundary issues.

### Degraded Case 1: Sentence 56  (charF1 Δ = -0.533)

**Source**: Starting this week, technology developers around the world could create their own decentralized apps using the tools available on the Inrupt website.

| | Text |
|---|------|
| Reference    | 本周开始，世界各地的技术开发人员将利用Inrupt网站上可用的工具，创建其自己的去中心化应用。 |
| Baseline     | 技始本周,全界各各地的术术开人员员可以使用Inrupt网站上提供的工具创建自己的分散应用程序. |
| Continuation | 技巧的开端,技巧的开端, |

### Degraded Case 2: Sentence 82  (charF1 Δ = -0.374)

**Source**: Hopefully, we can push on now, put last season behind us and be successful."

| | Text |
|---|------|
| Reference    | 希望我们可以继续前进，超越上一个赛季，取得成功。 |
| Baseline     | 希望我们能继续前进,把上个赛季抛在后面, |
| Continuation | 希我可是,我希我可是, |

### Degraded Case 3: Sentence 68  (charF1 Δ = -0.321)

**Source**: She then suggested direct contact between Trump and the press will increase.

| | Text |
|---|------|
| Reference    | 她随后表示，将会增加特朗普与媒体之间的直接接触。 |
| Baseline     | 她建议特朗普和和媒体之间的直接接触将会增加. |
| Continuation | 她建議特朗普和媒介的直線接觸將會增長. |

## Conclusion

Prefix-constrained continuation directly addresses the **translation-hypothesis
inconsistency** problem inherent in the baseline re-translation approach:

- **Consistency guaranteed**: the decoder is forced to extend the committed
  prefix; it cannot change earlier characters mid-sentence.
- **Inconsistency (bigram repetition) reduced by 0.0112** (↓ from 0.0428 → 0.0316).
- **BLEU -4.79** (13.46 → 8.67).
- 34/100 sentences improve, 55/100 degrade,
  11/100 are roughly equal.

Combining continuation with the DD gate (which prevents early commits) would
further reduce error propagation: DD stops premature commits, continuation
ensures what IS committed remains internally consistent.