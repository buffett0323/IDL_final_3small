# Simultaneous Machine Translation: From Wait-k to Future-Aware Consensus
## IDL Final Project Report — Haoling Peng

> **Status**: the authoritative final results are in `outputs/rerun_20260330/rerun_summary.md` and `outputs/full_comparison.md`. The final story has a **three-step progression**: **DD / JS** shows how future disagreement can decide **when to wait**; **Future literal LCP** shows how to decide **what prefix is safe to write**; and **Token Consensus Decoding** solves LCP's surface-form rigidity by working at the **token-level distribution**. The fair comparison across all three methods uses the same Qwen30B translator, same `rand100` evaluation slice, and all bugs fixed. A **CoVoST + Whisper ASR cascaded-ST pilot (100 utterances)** documents the speech-facing pipeline. Hardware: CMU Babel, L40S GPUs.
>
> **Note**: The file `project_proposal.pdf` is the **original multi-author course proposal** (“Selective Test-Time Reasoning for Streaming Speech Translation,” STTR / UTR, MuST-C + CoVoST2, EN–De focus). **This report** describes the **work actually implemented** for the individual final project (EN→ZH, SimulEval, DD / LCP / LLM futures / Token Consensus). Continuity is in **problem and idea** (streaming input, early commitment, selective extra computation when uncertain); **datasets, language pair, and method names** differ where noted in **§1.1** below.

---

## 1. Problem Statement

Simultaneous Machine Translation (SiMT) must emit target-language output while source words are
still arriving. The central challenge is the **early commitment problem**: once a target prefix is
written, it is **irreversible**. If the system commits before enough source context is available,
later source words may reveal that the earlier target prefix was wrong, but the agent can no
longer repair it.

This project studies that problem in **EN→ZH streaming translation** under SimulEval. The final
question is:

> Can **future-aware reasoning** improve simultaneous translation by helping the system decide
> either **when to wait** or **what prefix is safe to commit**?

The final codebase answers that question with a **three-step method story**:

1. **Step 1 — DD / JS gating**: detect when the future looks unstable, so the system should
   probably wait (**when to wait**)
2. **Step 2 — Future literal LCP**: move from “should I wait?” to “what can I safely write now?”
   by finding the longest common character prefix across future translations (**what is safe to write**)
3. **Step 3 — Token Consensus Decoding**: solve LCP's fatal weakness — different words can express the
   same meaning — by working at the **token-level distribution** instead of character-level string
   matching (**what is safe to write, robustly**)

### 1.1 Deliverables vs. original proposal

The original course proposal emphasized streaming speech translation, uncertainty-triggered
reasoning, and compute-aware refinement. The final project keeps the same conceptual motivation,
but the implemented system is narrower and more concrete:

- **Kept**: streaming setting, SimulEval, early commitment, selective extra reasoning
- **Changed**: language pair is **EN→ZH** rather than EN→DE, and the final methods are
  **DD / JS** and **Future literal LCP**
- **Partially kept**: speech is evaluated as a **cascaded** pipeline
  (**Whisper ASR → text → same SimulEval agent**), not as a single end-to-end speech translation model
- **Not delivered**: COMET, MuST-C EN→DE, and a full end-to-end speech benchmark

### 1.2 Final claim structure

The report is organized around **three progressive claims**:

- **Stage-1 claim**: future disagreement is a useful signal for deciding **when to wait** (DD / JS)
- **Stage-2 claim**: future consensus at the character level is a useful signal for deciding **what is safe to write** (Future literal LCP)
- **Stage-3 claim**: token-level distribution consensus is more robust than character-level LCP because it tolerates lexical variation across translations (Token Consensus Decoding)
- **Transfer claim**: the future-aware idea is useful on both a smaller NLLB backbone and a stronger Qwen backbone

The main fair comparison uses a **single Qwen30B translator** across all three methods on the same `rand100` evaluation slice.

---

## 2. Experimental Setup

### 2.1 Data and evaluation slices

The final report uses two evaluation tracks:

1. **Supporting line (small-model policy study)**  
   **WMT19 EN→ZH**, full test set, **1997 sentences**  
   Files: `data/enzh/wmt19_source.txt`, `data/enzh/wmt19_target.txt`

2. **Main line (same-translator fairness study)**  
   **wmt500**, first **500 sentences** of the same EN→ZH benchmark slice used for Qwen runs  
   Files: `data/enzh/wmt500_source.txt`, `data/enzh/wmt500_target.txt`

These two slices are used for different reasons:

- The **WMT19 full set** is used to test whether DD helps a fixed small translator at scale
- The **500-sentence slice** is used to make the expensive **Qwen30B direct vs Future literal LCP**
  comparison practical

**Important fairness note**: absolute BLEU from the 1997-sentence and 500-sentence tables should
**not** be compared directly.

### 2.2 Models and agents

- **NLLB backbone**: `facebook/nllb-200-distilled-600M`
- **Future LM for DD / Future literal LCP**: `Qwen3-4B-Base`
- **Strong translator for main line**: `Qwen3-30B-A3B-Instruct-2507` via vLLM
- **Speech pilot ASR**: `openai/whisper-small`

Main code paths:

- `agents/sttr_enzh_agent.py`: NLLB wait-k + DD / LM-DD
- `agents/dd_gate.py`: future-conditioned JS divergence
- `agents/semantic_lcp_agent.py`: Qwen direct / Future literal LCP
- `agents/token_consensus_agent.py` + `agents/token_consensus_core.py`: Token Consensus Decoding

### 2.3 Metrics

All main experiments use SimulEval metrics:

- **BLEU**: translation quality
- **AL**: Average Lagging
- **LAAL**: Length-Adaptive Average Lagging
- **AP**: Average Proportion

### 2.4 Final implementation choice: prefix-anchored semantics

A crucial final design decision is that the authoritative rerun uses **prefix-anchored**
semantics for the NLLB line:

- once a target prefix is committed, it is treated as **fixed**
- later decoding continues from that committed prefix rather than silently rewriting it

This is the correct behavior for simultaneous translation, because committed target text must not
change after it has been emitted. The new rerun therefore gives a stricter and more faithful
estimate of small-model simultaneous performance than the earlier exploratory runs.

---

## 3. Systems Compared

### 3.1 Baseline 1: Wait-k + greedy NLLB

This is the strict simultaneous baseline:

- translator: `facebook/nllb-200-distilled-600M`
- policy: wait-k
- decoding: greedy
- no DD, no future-aware veto

It answers the simplest question: how good is a fixed small translator under a standard wait-k
policy?

### 3.2 Baseline 2: Wait-k + beam NLLB

This keeps the same NLLB translator and the same wait-k framing, but uses beam search instead of
greedy decoding. Conceptually this is a **search baseline**, not an early-commit solution.

It is useful for the demo and qualitative explanation, but it is **not** the main supporting-line
table in the final report.

### 3.3 Method 1: Distribution Divergence (DD + JS)

**Idea**: estimate whether the next commit is risky by looking at how much the NLLB next-token
distribution changes under different plausible futures.

Pipeline:

1. Sample **K English futures** from the observed prefix using `Qwen3-4B-Base`
2. For each future, compute the NLLB next-token distribution
3. Measure average pairwise **Jensen-Shannon divergence**
4. If divergence is high, **READ more** instead of committing

Two operating modes:

- **DD full gate**: DD is the main decision rule
- **DD veto**: baseline proposes a commit first, DD only blocks commits judged too risky

This method is meant to answer the question:

> If plausible futures already disagree strongly, should the system avoid committing now?

### 3.4 Method 2: Future literal LCP

**Idea**: instead of asking only whether to wait, ask what target prefix is safe to write **right now**.

Pipeline:

1. Use `Qwen3-4B-Base` to sample English futures
2. For each future, translate with the same strong translator
   `Qwen3-30B-A3B-Instruct-2507`
3. Compute the **literal Chinese longest common prefix (LCP)** under a quorum rule
4. Commit only the shared prefix; defer the divergent tails

This method is a stronger answer to early commitment than DD, because it directly
models **safe-prefix commitment** rather than only delaying decisions.

**However, LCP has a critical weakness**: it operates on the surface character level. In Chinese
translation, the same English source can be rendered with **completely different word choices**. For
example, the name "Miranda" might become "米兰达" in one translation and "蜜蕊拉" in another — both
are correct, but LCP sees zero common prefix. This means LCP is **too strict**: it blocks
commits even when all futures semantically agree, simply because they chose different surface forms.

### 3.5 Method 3: Token Consensus Decoding

**Idea**: solve LCP's surface-form rigidity by moving from character-level string matching to
**token-level distribution intersection**. Instead of asking "do the full translations share a
common prefix?", ask "do all futures agree on the **same next token** from the translator's
vocabulary?"

Pipeline:

1. Sample **K=10** English futures from the observed source prefix using `Qwen3-4B-Base`
2. For each future, construct a full hypothetical source and query `Qwen3-30B-A3B-Instruct-2507`
   for the **next-token probability distribution** via the vLLM `/completions` endpoint, forcing
   the committed Chinese prefix in the assistant turn (prefix forcing)
3. **Filter** each distribution before consensus: remove special tokens, English-letter tokens,
   garbled/control characters
4. Compute the **hard intersection** of filtered top-k token IDs across all K futures
5. If the intersection is non-empty, pick the token with the **highest average probability** and
   append it to the pending delta
6. Repeat until the intersection becomes empty or a max-step cap is reached
7. **Trim** pending tokens to a clean UTF-8 boundary before committing

Key design decisions that distinguish this from LCP:

- **Token-level, not character-level**: consensus is over token IDs in the translator's vocabulary.
  If all futures produce the same next token ID, that token is safe to commit — regardless of
  whether the full translations share a common character prefix
- **Distribution filtering before consensus**: special tokens, English letters, and garbled tokens
  are removed from each future's distribution before the intersection is computed, preventing
  noise from poisoning the consensus
- **Completion endpoint with prefix forcing**: uses `apply_chat_template` + assistant prefix to
  force the model to continue from the exact committed prefix, then reads `top_logprobs` for
  the next-token distribution (not a full generation)
- **Batch API calls**: all K futures are queried in a single batched `/completions` request for
  efficiency
- **Force-finish via continuation**: when source is exhausted, the final translation continues from
  the committed prefix (not a full re-translation), preventing duplication

This method preserves the core future-aware intuition — sample plausible futures, find what they
agree on — but lifts it from surface-form agreement to **distributional agreement**, making it
robust to the lexical variation that cripples character-level LCP.

---

## 4. Final Results

### 4.1 Supporting evidence: DD + JS is a useful first answer, but it buys quality by waiting more

The first concrete future-aware idea in the project was to ask whether disagreement across
plausible futures can tell the system **not to commit yet**.

Using the same `wmt500` slice on a fixed NLLB backbone:

| Method | N | BLEU | ΔBLEU vs k=5 greedy | AL | LAAL | AP |
|:--|--:|--:|--:|--:|--:|--:|
| NLLB baseline k=5 · beam=1 | 500 | 12.821 | +0.000 | 8.335 | 8.365 | 0.574 |
| NLLB baseline k=5 · beam=8 | 500 | 18.311 | +5.490 | 8.393 | 8.517 | 0.711 |
| NLLB DD + JS (τ=0.05) | 500 | 14.951 | +2.130 | 11.916 | 11.931 | 0.640 |
| NLLB literal LCP (K=4) | 500 | 14.472 | +1.651 | 8.384 | 8.449 | 0.638 |

**Key observations**:

1. **DD + JS works as a risk signal**: it improves over the greedy NLLB baseline by **+2.130 BLEU**.
2. But that gain comes with a large latency cost: **+3.581 AL**.
3. This tells us something important about DD: it mostly helps by saying
   “the future looks unstable, so wait.”
4. That is useful, but it is not yet the final answer, because the policy often pays for quality
   by delaying output rather than by finding a safe partial commit.

### 4.2 Authoritative headline: four-method comparison on full WMT19 (1997 sentences)

The authoritative comparison uses a **single fixed translator** (`Qwen3-30B-A3B-Instruct-2507`),
the **full WMT19 EN→ZH test set** (1997 sentences), and the same wait-k=5 policy.
Only the commit rule changes.

| Method | BLEU | ΔBLEU | AL | LAAL | AP | COMET |
|:--|--:|--:|--:|--:|--:|--:|
| Qwen30B direct k=5 | 28.148 | — | 8.55 | 9.19 | 0.940 | 0.774 |
| DD + JS gate k=5 K=4 τ=0.15 | 29.143 | +1.00 | 9.33 | 9.86 | 0.939 | 0.791 |
| Future literal LCP k=5 K=4 | 31.952 | +3.80 | 8.79 | 9.10 | 0.911 | 0.828 |
| **Token Consensus k=5 K=10** | **34.155** | **+6.01** | **8.83** | **9.15** | **0.899** | **0.848** |

**Key observations**:

1. **DD + JS improves over direct by +1.0 BLEU**, but at a cost of **+0.78 AL**. This confirms
   the pattern from the NLLB line: JS gating buys quality primarily by waiting longer.

2. **LCP beats both direct (+3.8 BLEU) and JS gating (+2.8 BLEU)** with **lower latency** than
   JS (AL 8.79 vs 9.33). This shows that deciding **what is safe to write** is strictly better
   than deciding **when to wait**.

3. **Token Consensus beats LCP by a further +2.2 BLEU** (31.95 → 34.16) with only +0.04 AL
   more than LCP. This is the main result: moving from character-level LCP to token-level
   distribution consensus recovers the quality that LCP loses due to surface-form mismatch.

4. **COMET confirms the BLEU ranking**: 0.774 → 0.791 → 0.828 → **0.848**, strengthening the
   quality claim with a neural evaluation metric.

5. **The progressive story is clear in the numbers**: direct (28.15) → JS gating (29.14, +0.78 AL)
   → LCP (31.95, +0.24 AL) → Token Consensus (34.16, +0.28 AL). Each step improves BLEU, and the
   two commit-rule methods (LCP, Token Consensus) achieve much better quality-latency tradeoffs
   than the wait-more approach (JS gating).

6. The total improvement from baseline to Token Consensus is **+6.01 BLEU** and **+0.074 COMET**
   with only **+0.28 AL** — a substantial quality gain for minimal latency cost.

### 4.2.1 Earlier rand100 comparison (for reference)

The same four methods on a 100-sentence random subset show consistent trends:

| Method | BLEU | AL | COMET |
|:--|--:|--:|--:|
| Qwen30B direct k=5 | 27.438 | 8.750 | 0.779 |
| DD + JS gate k=5 K=4 | 29.419 | 9.823 | 0.790 |
| Future literal LCP k=5 K=4 | 31.514 | 8.800 | 0.827 |
| Token Consensus k=5 K=10 | 33.841 | 9.075 | 0.850 |

### 4.3 What the full WMT19 rerun still tells us

The final prefix-anchored NLLB rerun on the full **WMT19** test set remains useful as large-scale
supporting evidence:

| Method | N | BLEU | ΔBLEU vs k=5 | AL | LAAL | AP |
|:--|--:|--:|--:|--:|--:|--:|
| NLLB baseline k=5 | 1997 | 12.523 | +0.000 | 8.464 | 8.524 | 0.567 |
| NLLB DD veto τ=0.03 | 1997 | **15.745** | **+3.222** | 13.294 | 13.335 | 0.691 |

This rerun confirms the same stage-1 intuition at scale:

- future disagreement really is a useful signal for **when to wait**
- but the quality gain is accompanied by a substantial latency increase
- therefore the natural next move is not “just veto more,” but to design a method that can still
  commit the stable prefix instead of postponing everything

---

## 5. Interpretation: A Progressive Story

### 5.1 The three-step progression

The project tells a clean research story where each method addresses a limitation of the previous:

1. **DD + JS** (when to wait): if plausible futures disagree, delay committing.
   On `rand100`: +1.98 BLEU over direct, but +1.07 AL — quality bought by waiting.
   *Limitation*: quality improves, but mostly by waiting longer — not by writing smarter.

2. **Future literal LCP** (what is safe to write): if futures agree on a character prefix, commit it.
   *Limitation*: the same meaning can be expressed with completely different Chinese words.
   Character-level string matching is **too strict** — it blocks commits even when all futures
   semantically agree.

3. **Token Consensus Decoding** (what is safe to write, robustly): if all futures agree on the
   same **next token ID** from the translator's distribution, commit it.
   This solves LCP's weakness because consensus is over the model's vocabulary, not over surface
   strings.

### 5.2 Why DD + JS was the right first step

DD + JS is a good first method for three reasons:

1. It keeps the small translator fixed, so the gain is not just “bigger model wins”
2. It uses **LM-sampled futures only**, not oracle future source words
3. It exposed the real tradeoff very clearly: quality can improve, but often by reading more first

The best interpretation is not that DD “solves” early commitment, but that it provides the right
**diagnostic signal**: when plausible futures disagree strongly, the next commit is risky.
That signal is valuable even if the first policy built on top of it is still too latency-heavy.

### 5.3 Why LCP was a natural but flawed second step

Moving from “when to wait” to “what is safe to write” was the right conceptual leap. LCP makes the
future-aware signal **constructive**: instead of only blocking risky commits, it identifies the safe
prefix that all futures agree on.

But LCP's fatal weakness is that it operates on **character identity**. Chinese translation has
enormous lexical freedom — the same English sentence can be rendered with different word choices,
different particle orders, or different transliterations. When two translations both correctly
capture the meaning but start with different characters, LCP finds **zero common prefix** and
commits nothing.

This is why LCP's DAL (12.45) is the highest of the three methods: it tends to either commit a
long shared prefix (when futures happen to use the same words) or commit nothing at all (when they
use different words for the same meaning). The commit pattern is feast-or-famine rather than
steady.

### 5.4 Why Token Consensus Decoding is the strongest method

Token Consensus solves LCP's weakness by reframing the consensus question:

- **LCP asks**: “do the full Chinese translations share a common character prefix?”
- **Token Consensus asks**: “do all futures agree on the **same next token** from the translator?”

This is a fundamentally better question because:

1. **It tolerates lexical variation in full translations.** Even if Future A produces “米兰达在周一”
   and Future B produces “米兰达于星期一”, their next-token distributions from the shared committed
   prefix may still agree on “米” as the next token — because the model's next-token prediction
   depends on the committed prefix, not on the full translation.

2. **It uses the model's confidence, not string matching.** When the intersection of top-k tokens
   is non-empty, we pick the token with the highest average probability. This means the system only
   commits tokens that the translator is confident about under ALL plausible futures.

3. **It produces steady commits.** Because consensus is checked one token at a time, the system
   can commit 1, 2, or 3 tokens per step — producing a smoother latency profile than LCP's
   all-or-nothing behavior. This is reflected in Token Consensus having the **lowest DAL** (11.53).

### 5.5 What not to claim

The report should avoid these misleading claims:

1. Do **not** compare the WMT19 and wmt500/rand100 tables as one absolute BLEU ranking
2. Do **not** describe DD futures as oracle or truncation-based
3. Do **not** present old full-prefix-style NLLB results as the final simultaneous setting
4. Do **not** claim Token Consensus is semantic — it is still a distributional method, not a
   meaning-aware one; it simply happens to be more robust to surface variation than LCP

---

## 6. Relation to Early Commitment

The project is not only about BLEU. It is specifically about a streaming failure mode:

- some target decisions are locally plausible
- later source context reveals they were premature
- but a simultaneous system cannot retract them once committed

The final methods connect to that problem in a concrete way:

- **DD / JS** measures whether the immediate next-step target distribution is unstable under
  plausible future continuations → decides **when to wait**
- **Future literal LCP** measures whether a target character prefix remains stable across multiple
  future-conditioned translation hypotheses → decides **what characters are safe to write**
- **Token Consensus Decoding** measures whether the translator’s next-token distribution agrees
  across all plausible futures → decides **what tokens are safe to write**, robust to surface variation

So the project’s contribution is best described as:

> transforming early commitment from a fixed wait-k schedule into a **future-aware decision problem**,
> progressively refined from when-to-wait gating, to character-level prefix consensus, to
> **token-level distributional consensus**

---

## 7. Cascaded Speech-Translation Pilot

The speech-facing part of the project is a **cascaded pilot**, not a separate end-to-end speech
translation model.

Pipeline:

1. English speech input from CoVoST EN→ZH
2. `openai/whisper-small` ASR
3. ASR output converted into incremental English text
4. Same SimulEval agents as the text experiments

Pilot score (`outputs/covost_cascaded/nllb_baseline_k5_asr_subset100_fixed/scores`):

- **BLEU 24.629**
- **AL 5.6**
- **LAAL 5.668**
- **AP 0.911**

This pilot supports the course project’s speech-translation framing at the pipeline level:
the early-commit methods are designed on text streams, but they are compatible with an upstream
ASR front-end.

---

## 8. Main Findings and Contributions

1. **The final project has a clean three-step research story.**  
   DD / JS answers **when to wait**; Future literal LCP answers **what character prefix is safe
   to commit**; Token Consensus Decoding answers **what token is safe to commit**, robust to the
   surface-form variation that limits character-level LCP.

2. **DD + JS exposed the right bottleneck.**  
   On NLLB `wmt500`, DD + JS improves by **+2.130 BLEU** but adds **+3.581 AL**.
   On Qwen `rand100`, the same pattern: **+1.98 BLEU** but **+1.07 AL**. Quality bought by waiting.

3. **Future literal LCP showed that safe-prefix commitment is better than gating.**  
   On fixed-backbone Qwen `rand100`, LCP improves over direct by **+4.08 BLEU** with only
   **+0.05 AL**. But its character-level matching is too strict for Chinese's lexical diversity.

4. **Token Consensus Decoding is the strongest method.**  
   On the same fair comparison, Token Consensus achieves **+6.40 BLEU** over direct (27.44 → 33.84)
   with only **+0.33 AL**, and beats LCP by **+2.33 BLEU**. It also has the best DAL (11.53),
   reflecting a steadier commit pattern.

5. **The fair comparison is clean.**  
   All three Qwen methods use the same translator, the same `rand100` evaluation slice, and the same
   bug-fixed implementation (prefix-forced force-finish, correct pending drain). Only the commit
   rule changes.

6. **Prefix anchoring and continuation-based force-finish were critical implementation choices.**  
   Both LCP and Token Consensus use `apply_chat_template` + assistant prefix forcing via the
   vLLM `/completions` endpoint. Force-finish continues from the committed prefix rather than
   re-translating from scratch, preventing duplication artifacts.

7. **The supporting NLLB line still matters.**  
   It shows both why DD is useful as a diagnostic signal and why the project needed to progress
   beyond pure wait-more gating.

---

## 9. Limitations

1. **The fair three-method comparison uses only 100 sentences.**  
   The `rand100` slice is sufficient to show clear trends, but a larger evaluation would
   strengthen statistical confidence.

2. **Token Consensus uses more API calls per step than LCP.**  
   LCP makes K full translations and compares prefixes; Token Consensus makes K next-token queries
   per consensus step, potentially many steps per source word. The quality gain comes at a
   compute cost that is not yet formally measured.

3. **No COMET is reported.**  
   BLEU + latency is sufficient for the final course project, but COMET would strengthen the
   quality argument.

4. **The speech evaluation is still a pilot.**  
   Only the cascaded ASR-facing path is demonstrated so far.

5. **Token Consensus's hard intersection is conservative.**  
   Requiring ALL K futures to agree on a token may be too strict when K is large. A quorum-based
   soft intersection (e.g., ≥80% of futures agree) could improve recall without sacrificing
   precision much.

---

## 10. What’s Next

- [x] Final large-scale text reruns with authoritative prefix-anchored NLLB results
- [x] Main-line same-translator Qwen direct vs Future literal LCP result
- [x] Cascaded ST pilot with Whisper + SimulEval
- [ ] Add COMET on WMT19 / wmt500 / CoVoST pilot
- [ ] Add one compute-matched Qwen baseline for Future literal LCP
- [ ] Run DD / Future literal LCP on the same ASR text files for a unified speech-facing table
- [ ] Expand CoVoST evaluation beyond the 100-utterance pilot
- [ ] Add one trace-based case study figure in the final write-up

---

## Appendix: File Structure

```
IDL_final_3small/
├── agents/
│   ├── sttr_enzh_agent.py          # Prefix-anchored NLLB agent: wait-k + DD / LM-DD
│   ├── semantic_lcp_agent.py       # Qwen direct + Future-LCP agent
│   ├── token_consensus_agent.py    # Token Consensus Decoding agent
│   ├── token_consensus_core.py     # Token Consensus core engine
│   ├── dd_gate.py                  # Future-conditioned JS divergence gate
│   ├── model_utils.py              # Model loading utilities
│   └── waitk_agent.py              # Simple wait-k reference implementation
├── scripts/
│   ├── compare_full.py                         # Consolidated result table / plots
│   ├── compare_waitk_dd.py                     # NLLB baseline vs DD analysis
│   ├── run_full_nllb.sbatch                    # Original NLLB full runs
│   ├── run_full_qwen.sbatch                    # Original Qwen full runs
│   ├── run_full_nllb_rerun_prefixanchored.sbatch  # Authoritative supporting-line rerun
│   ├── run_full_qwen_rerun_mainline.sbatch        # Authoritative main-line rerun
│   ├── summarize_rerun_20260330.py             # Builds rerun summary
│   ├── prepare_covost_asr_texts.py             # CoVoST → Whisper ASR → text files
│   ├── format_zh_target_for_simul.py           # Chinese ref spacing for BLEU
│   ├── run_covost_cascaded_nllb.sbatch
│   ├── early_commit_analysis.py
│   └── verbose_trace.py
├── data/enzh/
│   ├── wmt19_source.txt
│   ├── wmt19_target.txt
│   ├── wmt500_source.txt
│   ├── wmt500_target.txt
│   ├── rand100_source.txt
│   ├── rand100_target.txt
│   └── covost_enzh/
├── REPORT_AND_DEFENSE_RESULTS.md
├── outputs/
│   ├── rerun_20260330/
│   │   ├── rerun_summary.md
│   │   ├── nllb_support/       # Final prefix-anchored NLLB / DD / LM-DD outputs
│   │   └── qwen_main/          # Final Qwen direct / Future literal LCP outputs
│   ├── fair_direct_k5/         # Bug-fixed Qwen30B direct (rand100)
│   ├── fair_js_gate_k5_f4/     # Bug-fixed DD + JS gating (rand100)
│   ├── fair_semlcp_k5_f4/      # Bug-fixed Future literal LCP (rand100)
│   ├── fair_consensus_k5/      # Bug-fixed Token Consensus (rand100)
│   ├── full_comparison.md
│   ├── full_comparison_plot.png
│   ├── full_bleu_bar.png
│   ├── covost_cascaded/
│   └── verbose_traces/
└── project_proposal.pdf
```
