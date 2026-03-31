# Simultaneous Machine Translation: From Wait-k to Future-Aware Consensus
## IDL Final Project Report — Haoling Peng

> **Status**: the authoritative final results are in `outputs/rerun_20260330/rerun_summary.md` and `outputs/full_comparison.md`. The final story now has a clear progression: **DD / JS** first shows how future disagreement can decide **when to wait**, and **Future literal LCP** then shows how to decide **what is still safe to write** without paying the same latency cost. The most presentation-friendly tables are the fixed-backbone `wmt500` comparisons; the prefix-anchored **full WMT19 rerun** remains the large-scale supporting evidence. A **CoVoST + Whisper ASR cascaded-ST pilot (100 utterances)** documents the speech-facing pipeline. Narration: **`REPORT_AND_DEFENSE_RESULTS.md`**. Hardware: CMU Babel, L40S GPUs.
>
> **Note**: The file `project_proposal.pdf` is the **original multi-author course proposal** (“Selective Test-Time Reasoning for Streaming Speech Translation,” STTR / UTR, MuST-C + CoVoST2, EN–De focus). **This report** describes the **work actually implemented** for the individual final project (EN→ZH, SimulEval, DD / LCP / LLM futures). Continuity is in **problem and idea** (streaming input, early commitment, selective extra computation when uncertain); **datasets, language pair, and method names** differ where noted in **§1.1** below.

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

The final codebase answers that question with a **two-step method story**:

1. **Step 1**: use **DD / JS gating** to detect when the future looks unstable, so the system should
   probably wait
2. **Step 2**: move from “should I wait?” to “what can I safely write now?” using
   **Future literal LCP**

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

The report is organized around **two different kinds of claims**, which should not be mixed:

- **Stage-1 claim**: future disagreement is a useful signal for deciding **when to wait**
- **Stage-2 claim**: future consensus is a useful signal for deciding **what is safe to write**
- **Transfer claim**: the future-aware idea is useful on both a smaller NLLB backbone and a stronger Qwen backbone

This distinction matters because the two lines use different translators and different evaluation
slices, so only **within-line** comparisons are fair.

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

This method is the cleanest answer to early commitment in the final project, because it directly
models **safe-prefix commitment** rather than only delaying decisions.

---

## 4. Final Large-Scale Results

### 4.1 Step 1: DD + JS is a useful first answer, but it buys quality by waiting more

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

### 4.2 Step 2: Future literal LCP turns the same intuition into a better commit rule

The next step is to keep the same future-aware intuition, but ask a stronger question:
if futures disagree, **what part is still stable enough to write now**?

On a fixed Qwen30B translator with the same `wmt500` slice:

| Method | N | BLEU | ΔBLEU vs direct | AL | LAAL | AP |
|:--|--:|--:|--:|--:|--:|--:|
| Qwen30B direct k=5 | 500 | 21.726 | +0.000 | 8.400 | 8.662 | 0.668 |
| Qwen30B DD + JS | 500 | 22.266 | +0.540 | 9.170 | 9.387 | 0.673 |
| Future literal LCP K=4 | 500 | **24.072** | **+2.346** | 8.678 | 8.897 | 0.658 |

**Key observations**:

1. This is the cleanest fairness comparison in the whole project: the translator is fixed and only
   the commit rule changes.
2. DD + JS still helps, but mostly as a conservative wait policy.
3. Future literal LCP is much stronger: **+2.346 BLEU** for only **+0.278 AL**.
4. This is the main headline result because it preserves the future-aware idea while avoiding the
   large latency penalty seen in DD-style gating.

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

## 5. Interpretation

### 5.1 What the two lines prove

The project ends up with a clean method progression:

1. **DD + JS** is the right first idea: future disagreement is a useful signal for deciding
   **when to wait**
2. Its weakness is also clear: it often buys quality mainly by delaying commits
3. **Future literal LCP** is the better second step: use the same future-aware information to decide
   **what is still safe to write now**

These are different but compatible answers to early commitment.

### 5.2 Why DD + JS was the right first step

DD + JS is a good first method for three reasons:

1. It keeps the small translator fixed, so the gain is not just “bigger model wins”
2. It uses **LM-sampled futures only**, not oracle future source words
3. It exposed the real tradeoff very clearly: quality can improve, but often by reading more first

The best interpretation is not that DD “solves” early commitment, but that it provides the right
**diagnostic signal**: when plausible futures disagree strongly, the next commit is risky.
That signal is valuable even if the first policy built on top of it is still too latency-heavy.

### 5.3 Why Future literal LCP is the better final method

Future literal LCP is the strongest method in the project because it addresses early commitment
more directly than DD:

- DD says: “this looks risky, maybe wait”
- Future literal LCP says: “these characters are agreed upon, commit only that safe prefix”

This is a more direct fit to the simultaneous setting, where the central question is not only
whether to delay, but exactly **how much target text is safe to emit right now**.

### 5.4 What not to claim

The report should avoid three misleading claims:

1. Do **not** compare the WMT19 and wmt500 tables as one absolute BLEU ranking
2. Do **not** describe DD futures as oracle or truncation-based
3. Do **not** present old full-prefix-style NLLB results as the final simultaneous setting

---

## 6. Relation to Early Commitment

The project is not only about BLEU. It is specifically about a streaming failure mode:

- some target decisions are locally plausible
- later source context reveals they were premature
- but a simultaneous system cannot retract them once committed

The final methods connect to that problem in a concrete way:

- **DD / JS** measures whether the immediate next-step target distribution is unstable under
  plausible future continuations
- **Future literal LCP** measures whether a target prefix remains stable across multiple
  future-conditioned translation hypotheses

So the project’s contribution is best described as:

> transforming early commitment from a fixed wait-k schedule into a **future-aware decision problem**

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

1. **The final project has a clean research story.**  
   DD / JS is the first answer to **when to wait**; Future literal LCP is the stronger answer to
   **what is safe to commit**.

2. **DD + JS exposed the right bottleneck.**  
   On fixed-backbone NLLB `wmt500`, DD + JS improves over greedy `wait-k=5` by **+2.130 BLEU**,
   but also adds **+3.581 AL**.

3. **Future literal LCP improves the quality-latency tradeoff.**  
   On fixed-backbone Qwen `wmt500`, Future literal LCP improves over direct translation by
   **+2.346 BLEU** with only **+0.278 AL**.

4. **Prefix anchoring was the correct final design choice.**  
   It makes the small-model line stricter and more faithful to simultaneous translation because
   committed target text can no longer be rewritten implicitly.

5. **The main final claim should come from the Qwen line, not from cross-model comparisons.**  
   The cleanest headline is the same-translator result:
   `Qwen30B direct` → `Future literal LCP`.

6. **The supporting NLLB line still matters.**  
   It shows both why DD is useful and why the project needed to move beyond pure wait-more gating.

---

## 9. Limitations

1. **The two main tables use different evaluation slices.**  
   This is why the report must emphasize within-line comparisons.

2. **No COMET is reported.**  
   BLEU + latency is sufficient for the final course project, but COMET would strengthen the
   quality argument.

3. **The speech evaluation is still a pilot.**  
   Only the cascaded ASR-facing path is demonstrated so far.

4. **Compute-matched Qwen ablations are still limited.**  
   The strongest missing comparison is an “always-K-candidate” or similar compute-matched upper bound.

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
│   ├── sttr_enzh_agent.py      # Prefix-anchored NLLB agent: wait-k + DD / LM-DD
│   ├── semantic_lcp_agent.py   # Qwen direct + Future-LCP agent
│   ├── dd_gate.py              # Future-conditioned JS divergence gate
│   ├── model_utils.py          # Model loading utilities
│   └── waitk_agent.py          # Simple wait-k reference implementation
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
│   ├── full_comparison.md
│   ├── full_comparison_plot.png
│   ├── full_bleu_bar.png
│   ├── covost_cascaded/
│   └── verbose_traces/
└── project_proposal.pdf
```
