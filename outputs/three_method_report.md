# Three-Way Comparison: Translation Strategies for EN→ZH SiMT

## Methods

| # | Method | Core Mechanism |
|---|--------|----------------|
| 1 | **NLLB baseline** | Re-translate full source prefix at each step; emit `translation[tgt_len]`. Causes hypothesis inconsistency. |
| 2 | **NLLB continuation** | Force NLLB decoder with committed prefix via `decoder_input_ids`. Eliminates inconsistency but suffers tokenisation boundary mismatch. |
| 3 | **Qwen continuation** | Append committed Chinese to few-shot prompt; causal LM generates the next token naturally. No encoder-decoder mismatch. |

## System-Level Scores

| Method | BLEU | AL | LAAL | AP | Mean charF1 | Mean inconsistency |
|--------|------|----|------|----|--------------|--------------------|
| NLLB baseline | 13.46 | 8.80 | 8.82 | 0.583 | 0.4101 | 0.0428 |
| NLLB continuation | 8.67 | 8.68 | 8.71 | 0.531 | 0.3726 | 0.0316 |
| Qwen continuation | 8.46 | 8.70 | 9.04 | 0.784 | 0.4002 | 0.0635 |

## Per-Sentence Outcome (vs NLLB Baseline)

| Method | Better (Δ>0.01) | Equal | Worse (Δ<-0.01) |
|--------|-----------------|-------|-----------------|
| NLLB continuation | 34 (34%) | 11 | 55 (55%) |
| Qwen continuation | 54 (54%) | 4 | 42 (42%) |

## Top 5 Sentences: Qwen Most Improved vs NLLB Baseline

### Case 1: Sent 25  (Qwen charF1 Δ vs baseline = +0.533)

**Source**: That has not happened yet.

| Method | Output | charF1 |
|--------|--------|--------|
| Reference       | 但是尚未宣布。 | — |
| NLLB baseline   | 这还没有发生. | 0.000 |
| NLLB cont.      | 这还没发现. | 0.000 |
| Qwen cont.      | 但那还尚未发生。 | 0.533 |

### Case 2: Sent 27  (Qwen charF1 Δ vs baseline = +0.429)

**Source**: Kanye West: Rapper changes his name to Ye

| Method | Output | charF1 |
|--------|--------|--------|
| Reference       | 坎耶·维斯特：说唱歌手将他的名字改为"Ye" | — |
| NLLB baseline   | 美国歌支持. | 0.071 |
| NLLB cont.      | 美高梅的新歌手, | 0.200 |
| Qwen cont.      | 侃爷：说唱歌手改名耶 | 0.500 |

### Case 3: Sent 69  (Qwen charF1 Δ vs baseline = +0.390)

**Source**: Trainers and medical personnel gave Abercrombie oxygen on the sideline before placing him on a stretcher and taking him back for further evaluation.

| Method | Output | charF1 |
|--------|--------|--------|
| Reference       | 培训师和医务人员在边线旁给阿伯克龙比输了氧气，然后将他抬到担架上，带回来进行进一步评估。 | — |
| NLLB baseline   | 培练人员和医疗人员在将Abercrombie放在床上,, | 0.222 |
| NLLB cont.      | 培训人和医疗人才在安德克伦比被带回床上进去进度评价之前, | 0.389 |
| Qwen cont.      | 体能训导员及医事人等在场边给阿伯克龙比喂氧，之后将他抬上担架送回医检室进一一评量。 | 0.612 |

### Case 4: Sent 49  (Qwen charF1 Δ vs baseline = +0.253)

**Source**: Attempted murder charge over Belfast restaurant stabbing

| Method | Output | charF1 |
|--------|--------|--------|
| Reference       | 贝尔法斯特一家餐厅发生蓄意持刀杀人案件 | — |
| NLLB baseline   | 关于让我们可以在Belfast餐厅里做好. | 0.100 |
| NLLB cont.      | 关键是,我不想让你知. | 0.000 |
| Qwen cont.      | 试图谋杀罪在贝尓法斯餐館刺傷案 | 0.353 |

### Case 5: Sent 79  (Qwen charF1 Δ vs baseline = +0.244)

**Source**: That seat had been held by a Republican for over a decade, and President Donald Trump won the district by 20 points.

| Method | Output | charF1 |
|--------|--------|--------|
| Reference       | 之前，这一席位由共和党人稳坐了十多年，而唐纳德·特朗普总统以20个百分点的优势在这个选区获胜。 | — |
| NLLB baseline   | 座在解人曾经了这了席位十多年,年, | 0.281 |
| NLLB cont.      | 座位被共产党人占据了十多年, | 0.262 |
| Qwen cont.      | 该座位一由共党党派把守超十载，而唐纳德特朗普总长以二十点优胜此区。 | 0.525 |

## Sentences Where Qwen Continuation Degrades

### Degraded Case 1: Sent 82  (Δ = -0.461)

**Source**: Hopefully, we can push on now, put last season behind us and be successful."
| | Output |
|---|--------|
| Reference      | 希望我们可以继续前进，超越上一个赛季，取得成功。 |
| NLLB baseline  | 希望我们能继续前进,把上个赛季抛在后面, |
| Qwen cont.     | 也希望我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我我 |

### Degraded Case 2: Sent 85  (Δ = -0.454)

**Source**: "We think the decision to seek the death penalty rather than accepting a guilty plea to life in prison with no possibility of release will only prolong the trauma of these events for everyone involved," Patton said.
| | Output |
|---|--------|
| Reference      | 巴顿说：“我们认为，决定寻求死刑，而不去接受终身监禁而不可能获释的认罪，只会延长这些事件对所有相关人员的创伤。” |
| NLLB baseline  | "我们认为,寻求死刑的决定,而不是接受,名承认可能释放的终身监禁, |
| Qwen cont.     | 他觉察到，自1980年以降，美利坚合众国的经融体例已发作过两回重大变乱。第两回是2007-2008年环抱次级抵偿品的金流危急；第三回则是2015年环抱欧罗巴区债市危急。 |

### Degraded Case 3: Sent 97  (Δ = -0.450)

**Source**: Pyongyang has made no public comments.
| | Output |
|---|--------|
| Reference      | 朝鲜没有公开发表评论。 |
| NLLB baseline  | 平没有公开评论. |
| Qwen cont.     | 朝鮮平壤未發言論評價。 |

## Discussion

### Why NLLB continuation falls short

NLLB uses a seq2seq architecture: the encoder processes the source, the decoder
generates the target starting from `[decoder_start, lang_token]`. When we force
`decoder_input_ids = [decoder_start, zho_Hans, *committed_tokens]`, the model
must continue from an ARBITRARY Chinese prefix that was generated from a SHORTER
source prefix. The decoder cannot resolve the ambiguity of a single character
prefix (e.g., "美" could be the start of "美国", "美联储", "美好", etc.).
Result: lower BLEU than baseline despite reduced inconsistency.

### Why Qwen continuation is more robust

A causal LM processes source AND committed target as a unified token sequence.
The prompt ending with the committed Chinese text provides unambiguous context:
the model has seen `English: ... Chinese: 美国的` and knows exactly what was
meant, then continues naturally. There is no encoder-decoder context gap.

### Remaining limitations of Qwen base model

1. **No fine-tuning**: Qwen3-4B-Base was not fine-tuned for SiMT. It may output
   mixed Simplified/Traditional Chinese.
2. **Early-commit still happens**: wait-k determines when to start writing;
   combining Qwen continuation with DD gate would further reduce early-commit risk.
3. **Speed**: each character requires one forward pass through 4B parameters.
   Caching the static prompt prefix with KV cache would reduce this overhead.