# Implementation Plan: Selective Test-Time Reasoning for Streaming Speech Translation

## Executive Summary

This document provides a detailed, step-by-step implementation plan for the STTR project. It covers architecture decisions, code-level guidance, feasibility analysis for compute constraints, fallback strategies, and concrete deliverables for each phase. The plan is designed for a team of three (Jeng-Yue, Wilson, Haoling) working over 8 weeks with limited GPU access.

---

## 1. Architecture Decision: Cascaded vs. End-to-End

The first and most consequential decision is whether to build an **end-to-end** speech translation system or a **cascaded** (ASR → MT) pipeline.

### Recommendation: Cascaded Pipeline

For this project, a cascaded architecture is strongly recommended for the following reasons:

- **No training from scratch required.** End-to-end simultaneous ST models (e.g., those based on Fairseq S2T) require training on large ST corpora with multiple GPUs for days. A cascaded system lets you use pretrained ASR and MT models off the shelf.
- **Easier to isolate the research contribution.** The novelty of this project is the selective refinement mechanism, not the base translation model. A cascaded pipeline lets you plug in your uncertainty estimation and refinement logic at the MT stage without needing to modify a monolithic end-to-end model.
- **Industry-relevant.** Recent IWSLT 2024 and 2025 submissions show that cascaded systems (Whisper + NMT/LLM) remain competitive with or outperform end-to-end systems, especially when strong pretrained components are available.
- **Debugging is easier.** You can inspect the ASR output independently from the MT output, which is critical for understanding whether errors originate from recognition or translation.

### Proposed Architecture

```
Audio Stream → [Whisper ASR] → English text chunks → [wait-k NMT Agent] → German text
                                                            ↓
                                                   [Uncertainty Gate]
                                                      ↓ high        ↓ low
                                              [Refinement Pass]   [Commit directly]
```

**ASR component:** OpenAI Whisper (medium or large-v3, depending on GPU memory)
**MT component:** A pretrained En→De NMT model (Helsinki-NLP/opus-mt-en-de from HuggingFace, or facebook/mbart-large-50-many-to-many-mmt, or NLLB-200-distilled-600M)
**Simultaneous policy:** wait-k applied at the text-to-text level after ASR
**Evaluation:** SimulEval toolkit

---

## 2. Compute and Resource Feasibility

### What You Likely Have Access To

| Resource | Typical Academic Setup | Notes |
|----------|----------------------|-------|
| GPU | 1-4× NVIDIA A100/A6000/V100 (shared cluster) | Shared with other students; job queues |
| GPU hours | ~200-500 hours total over 8 weeks | Varies by institution |
| RAM | 32-64 GB | Per node |
| Storage | 500 GB - 1 TB | WMT test sets are <1 MB (text only) |

### Compute Budget Breakdown

| Task | Estimated GPU Hours | GPU Type |
|------|-------------------|----------|
| NMT baseline inference on WMT14 En-De (3003 sentences, all wait-k configs) | 1-2 hours | Single V100/A100 |
| Uncertainty estimation (entropy computation per token) | ~0 extra (computed during inference) | — |
| Multi-candidate generation (K=4-16 candidates) | 8-20 hours | Single V100/A100 |
| Refinement pass experiments (second-pass decoding) | 10-30 hours | Single V100/A100 |
| Ablation sweeps (τ threshold, K values, wait-k values) | 20-50 hours | Single V100/A100 |
| **Total estimate** | **~50-120 GPU hours** | |

This is very manageable. The key insight is that **you are not training any models** — all compute goes to inference and evaluation. This is the major advantage of the cascaded approach with pretrained models.

### What If Compute Is Extremely Limited (CPU-only or Colab)?

If you only have access to Google Colab or a laptop:

- Use **Whisper small** or **Whisper base** instead of large-v3 (4-10x faster, slight quality drop)
- Use **Helsinki-NLP/opus-mt-en-de** (~300M params) which runs comfortably on CPU
- Use **NLLB-200-distilled-600M** as a slightly larger alternative that still fits in Colab's free GPU
- WMT14 test set is only 3003 sentences (text-only), so compute is minimal
- Can also evaluate on WMT19 (1997 sentences) for a second data point

---

## 3. Phase-by-Phase Implementation

### Phase 1: Environment Setup and Data Pipeline (Week 1)

**Owner:** Jeng-Yue (data pipeline), Wilson (model setup)

#### 3.1.1 Environment Setup

```bash
# Create conda environment
conda create -n sttr python=3.10
conda activate sttr

# Core dependencies
pip install torch torchaudio transformers
pip install openai-whisper  # or pip install faster-whisper for speed
pip install sentencepiece sacrebleu
pip install simuleval  # from PyPI, or clone from GitHub for latest

# Optional but recommended
pip install unbabel-comet  # for COMET metric
pip install pandas matplotlib seaborn  # for analysis
```

#### 3.1.2 Download and Prepare WMT Test Data

WMT newstest is the standard MT benchmark. We use **WMT14 En→De** (3003 sentences) as the primary test set, downloaded automatically via sacrebleu with no authentication needed.

```bash
# Download WMT14 En→De and save in SimulEval format
python scripts/download_wmt.py

# Or download multiple test sets
python scripts/download_wmt.py --test-sets wmt14 wmt19
```

This produces:
```
data/wmt/
├── wmt14.json              # Full records (id, sentence, translation)
├── wmt14_source.txt        # One English sentence per line (SimulEval input)
└── wmt14_target.txt        # One German reference per line (SimulEval input)
```

Since we are using pretrained models (no training), the dataset is **evaluation-only**. WMT14 is the most widely cited En→De benchmark, making our results directly comparable to published work.

#### 3.1.3 Prepare SimulEval Input Format

SimulEval expects a source file (one source per line) and a target reference file. The download script already produces these:

```bash
# These are ready to use directly with SimulEval:
data/wmt/wmt14_source.txt   # English source sentences
data/wmt/wmt14_target.txt   # German reference translations
```

**Deliverables for Phase 1:**
- Working environment with all dependencies
- WMT14 En-De test set downloaded and formatted
- SimulEval source/target files ready for evaluation

---

### Phase 2: Baseline System — Wait-k with SimulEval (Week 2)

**Owner:** Wilson (SimulEval agent), Jeng-Yue (metric verification)

#### 3.2.1 Understanding the SimulEval Agent Interface

SimulEval uses an agent-based architecture. You implement a `TextToTextAgent` (or `SpeechToTextAgent`) that defines the simultaneous policy. The agent has two key methods:

- `policy()` — decides whether to READ more input or WRITE output
- The agent processes input incrementally and emits translations token by token

```python
# agents/waitk_agent.py
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from transformers import MarianMTModel, MarianTokenizer

class WaitKAgent(TextToTextAgent):
    """
    Wait-k simultaneous translation agent.
    
    Reads k source words before starting to translate,
    then alternates: read 1 source word, write 1 target word.
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.wait_k = args.wait_k
        self.model_name = "Helsinki-NLP/opus-mt-en-de"
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.model.eval()
        if args.device == "cuda":
            self.model.cuda()
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-k", type=int, default=5)
        parser.add_argument("--device", type=str, default="cuda")
    
    def policy(self):
        # Number of source words read so far
        src_len = len(self.states.source)
        # Number of target words written so far
        tgt_len = len(self.states.target)
        
        # If source is not finished and we haven't read k words yet, READ
        if not self.states.source_finished and src_len - tgt_len < self.wait_k:
            return ReadAction()
        
        # Otherwise, generate next target token(s) from current source prefix
        source_prefix = " ".join(self.states.source)
        translation = self._translate_prefix(source_prefix)
        
        # Extract the next word to emit
        translated_words = translation.split()
        if tgt_len < len(translated_words):
            next_word = translated_words[tgt_len]
            return WriteAction(next_word, finished=(
                self.states.source_finished and tgt_len + 1 >= len(translated_words)
            ))
        elif self.states.source_finished:
            return WriteAction("", finished=True)
        else:
            return ReadAction()
    
    def _translate_prefix(self, source_text):
        """Translate the current source prefix."""
        inputs = self.tokenizer(source_text, return_tensors="pt", padding=True)
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                num_beams=4,
                max_length=512,
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
```

#### 3.2.2 Running SimulEval

```bash
simuleval \
    --source source.txt \
    --target target.txt \
    --agent agents/waitk_agent.py \
    --wait-k 5 \
    --device cuda \
    --output outputs/baseline_k5/
```

SimulEval will automatically compute: BLEU, AL (Average Lagging), LAAL, DAL, and AP.

#### 3.2.3 Sweep wait-k Values

Run for k ∈ {3, 5, 7, 9} to establish the baseline quality-latency curve.

```bash
for k in 3 5 7 9; do
    simuleval \
        --source source.txt \
        --target target.txt \
        --agent agents/waitk_agent.py \
        --wait-k $k \
        --output outputs/baseline_k${k}/
done
```

#### 3.2.4 Sanity Check: Compare Against Known Results

Expected ballpark numbers for wait-k on WMT14 En-De (text-to-text, opus-mt):

| wait-k | BLEU (approx) | AL (approx) |
|--------|---------------|-------------|
| k=3    | 18-22         | ~3 words    |
| k=5    | 22-26         | ~5 words    |
| k=7    | 24-28         | ~7 words    |
| k=9    | 25-29         | ~9 words    |

If your numbers are wildly off, debug before proceeding. Common issues: tokenization mismatch, sentencepiece vs. word-level wait-k, detokenization before BLEU.

**Deliverables for Phase 2:**
- Working wait-k SimulEval agent with HuggingFace NMT model
- Baseline BLEU/AL/LAAL numbers for k ∈ {3, 5, 7, 9}
- Quality-latency curve plot (BLEU vs. AL)

---

### Phase 3: Multi-Candidate Selection — Method A (Weeks 3-4)

**Owner:** Wilson (multi-candidate decoding), Jeng-Yue (compute-matched baselines)

#### 3.3.1 Generating Multiple Candidates

At each commit point, instead of taking the single best beam search output, generate K diverse candidates using:

**Option 1: Diverse beam search**
```python
outputs = model.generate(
    **inputs,
    num_beams=K * 2,
    num_return_sequences=K,
    diversity_penalty=1.0,
    num_beam_groups=K,
    max_length=512,
)
```

**Option 2: Temperature sampling**
```python
candidates = []
for _ in range(K):
    output = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        max_length=512,
    )
    candidates.append(output)
```

#### 3.3.2 Re-ranking Candidates

Select the best candidate from the K options. Several strategies:

1. **Log-probability re-ranking** (simplest): Pick the candidate with the highest sequence-level log probability. This is already available from the model's `generate()` method when `output_scores=True`.

2. **MBR (Minimum Bayes Risk) decoding**: Pick the candidate that has the highest average similarity to all other candidates (measured via BLEU or COMET). More expensive but better quality.

3. **Quality Estimation re-ranking**: Use a reference-free QE model (e.g., COMET-QE) to score each candidate without needing the reference. This is the most principled approach but adds another model to the pipeline.

```python
# Simple log-prob re-ranking
def select_best_candidate(model, tokenizer, source_text, K=4):
    inputs = tokenizer(source_text, return_tensors="pt", padding=True).to(model.device)
    
    outputs = model.generate(
        **inputs,
        num_beams=K * 2,
        num_return_sequences=K,
        output_scores=True,
        return_dict_in_generate=True,
        max_length=512,
    )
    
    # outputs.sequences_scores contains log-probs for each sequence
    best_idx = outputs.sequences_scores.argmax()
    best_translation = tokenizer.decode(
        outputs.sequences[best_idx], skip_special_tokens=True
    )
    return best_translation, outputs.sequences_scores
```

#### 3.3.3 Compute-Matched Larger-Beam Baseline

This is a critical control. If Method A uses K=4 candidates, you need a baseline that uses beam_size=K*2=8 (roughly matching total compute). The point is to show that your multi-candidate + re-ranking approach is smarter than just throwing more beams at the problem.

```bash
# Larger-beam baseline
simuleval \
    --source source.txt \
    --target target.txt \
    --agent agents/waitk_agent.py \
    --wait-k 5 \
    --beam-size 8 \
    --output outputs/larger_beam_k5/
```

#### 3.3.4 Vary K

Test K ∈ {2, 4, 8, 16} and plot BLEU vs. compute (measured as wall-clock time or FLOPs).

**Deliverables for Phase 3:**
- Multi-candidate SimulEval agent
- Results for K ∈ {2, 4, 8, 16} with re-ranking
- Compute-matched larger-beam baseline
- Plot: BLEU vs. K, with compute-matched baselines overlaid

---

### Phase 4: Uncertainty Estimation (Weeks 5-6)

**Owner:** Haoling (uncertainty metrics), Wilson (integration)

This is the most research-critical phase. You need to build a module that, at each commit point, produces a scalar uncertainty score.

#### 3.4.1 Token-Level Entropy (Primary Signal)

The simplest and most common uncertainty measure. At each generated token, compute the entropy of the softmax distribution over the vocabulary.

```python
import torch
import torch.nn.functional as F

def compute_token_entropy(model, tokenizer, source_text):
    """
    Compute per-token entropy during greedy/beam decoding.
    Returns: list of (token, entropy) pairs.
    """
    inputs = tokenizer(source_text, return_tensors="pt", padding=True).to(model.device)
    
    # Run generation with output_scores=True to get logits at each step
    outputs = model.generate(
        **inputs,
        max_length=512,
        output_scores=True,
        return_dict_in_generate=True,
    )
    
    token_entropies = []
    for step_logits in outputs.scores:
        # step_logits shape: (batch_size, vocab_size)
        probs = F.softmax(step_logits[0], dim=-1)
        entropy = -(probs * probs.log()).sum().item()
        token_entropies.append(entropy)
    
    tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0])
    
    return list(zip(tokens, token_entropies))


def compute_sequence_uncertainty(token_entropies):
    """
    Aggregate token-level entropies into a sequence-level uncertainty score.
    Several options:
    """
    entropies = [e for _, e in token_entropies]
    
    return {
        "mean_entropy": sum(entropies) / len(entropies),
        "max_entropy": max(entropies),
        "sum_entropy": sum(entropies),
        "top3_mean": sum(sorted(entropies, reverse=True)[:3]) / 3,
    }
```

#### 3.4.2 Sequence-Level Log-Probability

The total log-probability of the generated sequence, normalized by length. Lower (more negative) values indicate higher uncertainty.

```python
def compute_sequence_logprob(model, tokenizer, source_text):
    inputs = tokenizer(source_text, return_tensors="pt", padding=True).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=512,
        output_scores=True,
        return_dict_in_generate=True,
    )
    
    # Compute log-prob of each selected token
    log_probs = []
    for i, step_logits in enumerate(outputs.scores):
        token_id = outputs.sequences[0, i + 1]  # +1 because sequences includes BOS
        log_prob = F.log_softmax(step_logits[0], dim=-1)[token_id].item()
        log_probs.append(log_prob)
    
    # Length-normalized log probability
    return sum(log_probs) / len(log_probs)
```

#### 3.4.3 MC Dropout Variance (Optional, More Expensive)

Run the model multiple times with dropout enabled at inference time. The variance across runs indicates epistemic uncertainty.

```python
def compute_mc_dropout_variance(model, tokenizer, source_text, n_runs=5):
    """MC Dropout: enable dropout at test time, run multiple forward passes."""
    model.train()  # enables dropout
    
    translations = []
    for _ in range(n_runs):
        inputs = tokenizer(source_text, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=512)
        translations.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    model.eval()  # disable dropout again
    
    # Measure agreement: if all runs produce the same output, low uncertainty
    unique_translations = set(translations)
    diversity_score = len(unique_translations) / n_runs
    
    return diversity_score, translations
```

Note: MC Dropout is 5x more expensive (n_runs forward passes). Only use if token entropy doesn't correlate well with errors.

#### 3.4.4 Correlation Analysis: Does Uncertainty Predict Errors?

Before building the gating mechanism, you MUST verify that high uncertainty actually corresponds to bad translations. This is the most important diagnostic in the entire project.

```python
# For each segment in the dev set:
# 1. Compute uncertainty score
# 2. Compute segment-level BLEU (or binary: good/bad based on BLEU threshold)
# 3. Measure correlation

import numpy as np
from scipy.stats import pearsonr, spearmanr

def correlation_analysis(uncertainties, bleu_scores):
    """
    Check if uncertainty predicts translation quality.
    High uncertainty should correlate with LOW bleu.
    """
    r_pearson, p_pearson = pearsonr(uncertainties, bleu_scores)
    r_spearman, p_spearman = spearmanr(uncertainties, bleu_scores)
    
    print(f"Pearson r={r_pearson:.3f} (p={p_pearson:.4f})")
    print(f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.4f})")
    
    # Also compute: if we threshold at various τ,
    # what fraction of "high uncertainty" segments are actually bad?
    for quantile in [0.5, 0.7, 0.8, 0.9]:
        threshold = np.quantile(uncertainties, quantile)
        high_unc_mask = np.array(uncertainties) > threshold
        
        avg_bleu_high = np.mean(np.array(bleu_scores)[high_unc_mask])
        avg_bleu_low = np.mean(np.array(bleu_scores)[~high_unc_mask])
        
        print(f"τ at {quantile:.0%}: "
              f"High-unc BLEU={avg_bleu_high:.1f}, "
              f"Low-unc BLEU={avg_bleu_low:.1f}, "
              f"Gap={avg_bleu_low - avg_bleu_high:.1f}")
```

**If the correlation is weak (<0.2 Spearman):** This is a significant risk flag. Consider:
- Switching to a different uncertainty signal (e.g., MC dropout, QE-based)
- Using the *max* token entropy rather than *mean* (spiky uncertainty often matters more)
- Computing uncertainty on the prefix-level rather than full-sequence level
- The model may be "confidently wrong" — this is acknowledged in your proposal's risks section

**Deliverables for Phase 4:**
- Uncertainty estimation module with at least 2 signals (entropy + log-prob)
- Correlation analysis on dev set with plots
- Written analysis: which uncertainty signal best predicts errors?
- Decision on which signal to use for the gating mechanism

---

### Phase 5: Selective Refinement — Methods B and C (Weeks 7-8)

**Owner:** Wilson (refinement integration), Haoling (ablation experiments), Jeng-Yue (error analysis)

#### 3.5.1 Method B: Always-Refine (Quality Upper Bound)

Apply a second-pass refinement to every commit point. This gives you the best possible quality at the cost of maximum latency — it serves as the upper bound.

**Refinement Strategy 1: Re-translation with full context**

At each commit point, instead of generating from the source prefix alone, also condition on the draft translation from the previous step.

```python
def refine_translation(model, tokenizer, source_prefix, draft_translation):
    """
    Refine by translating the source prefix again, 
    this time providing the draft as a prompt/prefix for the decoder.
    """
    # Option A: Simply re-translate with a larger beam
    inputs = tokenizer(source_prefix, return_tensors="pt", padding=True).to(model.device)
    refined = model.generate(
        **inputs,
        num_beams=8,
        max_length=512,
    )
    return tokenizer.decode(refined[0], skip_special_tokens=True)
```

**Refinement Strategy 2: Multi-candidate + re-rank (reuse Method A)**

Generate K candidates and re-rank — the same as Method A but applied at every commit point.

**Refinement Strategy 3: LLM-based post-editing (if resources allow)**

Use a small LLM (e.g., Gemma-2B, Phi-3-mini, or even an API call) to post-edit the draft.

```python
# This is more expensive but potentially powerful
prompt = f"""The following is a simultaneous translation from English to German. 
The translator only had partial context, so the translation may contain errors.

Source (partial): {source_prefix}
Draft translation: {draft_translation}

Please correct any errors in the draft translation. Output only the corrected German text."""
```

Note: LLM-based refinement adds significant latency. Only feasible if you have API access or a small local model. For the project, Strategy 1 or 2 is recommended.

#### 3.5.2 Method C: Uncertainty-Triggered Refinement (Main Contribution)

This is the core STTR system. The gate decides per-commit-point whether to refine.

```python
# agents/sttr_agent.py
class STTRAgent(TextToTextAgent):
    """
    Selective Test-Time Reasoning agent.
    Uses uncertainty gating to decide whether to refine each commit point.
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.wait_k = args.wait_k
        self.tau = args.uncertainty_threshold
        self.refinement_method = args.refinement_method  # "rerank" or "retranslate"
        self.K = args.num_candidates
        
        # Load models
        self.model = ...  # NMT model
        self.tokenizer = ...
        
        # Tracking
        self.num_refined = 0
        self.num_total = 0
    
    def policy(self):
        src_len = len(self.states.source)
        tgt_len = len(self.states.target)
        
        if not self.states.source_finished and src_len - tgt_len < self.wait_k:
            return ReadAction()
        
        source_prefix = " ".join(self.states.source)
        
        # Step 1: Generate draft translation
        draft, uncertainty = self._translate_with_uncertainty(source_prefix)
        
        self.num_total += 1
        
        # Step 2: Gate — refine if uncertain
        if uncertainty > self.tau:
            self.num_refined += 1
            translation = self._refine(source_prefix, draft)
        else:
            translation = draft
        
        # Step 3: Emit next word
        words = translation.split()
        if tgt_len < len(words):
            return WriteAction(words[tgt_len], finished=(
                self.states.source_finished and tgt_len + 1 >= len(words)
            ))
        elif self.states.source_finished:
            return WriteAction("", finished=True)
        else:
            return ReadAction()
    
    def _translate_with_uncertainty(self, source_text):
        """Generate translation and compute uncertainty score."""
        inputs = self.tokenizer(source_text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            num_beams=4,
            max_length=512,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        # Compute mean token entropy as uncertainty
        entropies = []
        for step_logits in outputs.scores:
            probs = F.softmax(step_logits[0], dim=-1)
            entropy = -(probs * probs.log()).sum().item()
            entropies.append(entropy)
        
        uncertainty = sum(entropies) / len(entropies) if entropies else 0.0
        translation = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        return translation, uncertainty
    
    def _refine(self, source_text, draft):
        """Apply refinement: re-rank K candidates."""
        inputs = self.tokenizer(source_text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            num_beams=self.K * 2,
            num_return_sequences=self.K,
            output_scores=True,
            return_dict_in_generate=True,
            max_length=512,
        )
        
        best_idx = outputs.sequences_scores.argmax()
        return self.tokenizer.decode(outputs.sequences[best_idx], skip_special_tokens=True)
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-k", type=int, default=5)
        parser.add_argument("--uncertainty-threshold", type=float, default=2.0)
        parser.add_argument("--refinement-method", type=str, default="rerank")
        parser.add_argument("--num-candidates", type=int, default=4)
```

#### 3.5.3 Threshold Calibration

Sweep τ on the dev set to find the Pareto-optimal operating points.

```bash
for tau in 0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0; do
    simuleval \
        --source dev_source.txt \
        --target dev_target.txt \
        --agent agents/sttr_agent.py \
        --wait-k 5 \
        --uncertainty-threshold $tau \
        --num-candidates 4 \
        --output outputs/sttr_tau${tau}/
done
```

Plot: BLEU vs. LAAL for each τ, overlaid with the baseline and always-refine curves.

#### 3.5.4 Refinement Trigger Rate Analysis

For each τ, report what fraction of commit points triggered refinement. This is a key efficiency metric.

| τ value | Trigger Rate | BLEU | LAAL | Notes |
|---------|-------------|------|------|-------|
| 0.0     | 100%        | ?    | ?    | = Always-Refine |
| 1.0     | ~80%        | ?    | ?    | |
| 2.0     | ~50%        | ?    | ?    | |
| 3.0     | ~20%        | ?    | ?    | |
| ∞       | 0%          | ?    | ?    | = Baseline |

The ideal operating point achieves most of the BLEU gain of always-refine while only triggering on 20-40% of commit points.

**Deliverables for Phase 5:**
- Always-Refine system (Method B) with results
- STTR agent (Method C) with uncertainty gating
- τ sweep results and Pareto frontier plot
- Trigger rate analysis table

---

### Phase 6: Ablation Studies and Error Analysis (Week 8)

**Owner:** Haoling (ablations), Jeng-Yue (error analysis), Wilson (writing)

#### 3.6.1 Ablation: Uncertainty Signal Comparison

Compare at least two uncertainty signals:

| Signal | Computation Cost | Expected Correlation |
|--------|-----------------|---------------------|
| Mean token entropy | Free (during decoding) | Moderate |
| Max token entropy | Free | Potentially better for spiky errors |
| Sequence log-prob | Free | Moderate |
| MC Dropout (n=5) | 5× cost | Higher but expensive |

#### 3.6.2 Ablation: Wait-k × Selective Refinement Interaction

Does selective refinement help more at low latency (small k) or high latency (large k)?

Run the STTR agent at the best τ for each k ∈ {3, 5, 7, 9}.

Hypothesis: The benefit should be largest at small k, because that's where early commitment errors are most frequent.

#### 3.6.3 Linguistically-Driven Error Analysis

Categorize errors in the baseline and STTR outputs by type. This can be done semi-automatically:

```python
# Heuristic error detection patterns for En-De
def detect_error_categories(source, reference, hypothesis):
    errors = {}
    
    # Negation: check if "nicht", "kein", "nie" etc. are present/absent
    neg_words_de = {"nicht", "kein", "keine", "keinen", "keinem", "nie", "niemals"}
    ref_has_neg = bool(neg_words_de & set(reference.lower().split()))
    hyp_has_neg = bool(neg_words_de & set(hypothesis.lower().split()))
    if ref_has_neg != hyp_has_neg:
        errors["negation"] = True
    
    # Numerals: check if numbers match
    import re
    ref_nums = set(re.findall(r'\d+', reference))
    hyp_nums = set(re.findall(r'\d+', hypothesis))
    if ref_nums != hyp_nums:
        errors["numeral"] = True
    
    # Named entities: simple check for capitalized words (imperfect but useful)
    ref_entities = set(w for w in reference.split() if w[0].isupper() and len(w) > 1)
    hyp_entities = set(w for w in hypothesis.split() if w[0].isupper() and len(w) > 1)
    if ref_entities - hyp_entities:
        errors["named_entity"] = True
    
    return errors
```

For **reordering errors**, a more sophisticated approach is needed. You can approximate this by computing word alignment (using tools like `awesome-align`) and checking if the alignment crosses are significantly different between reference and hypothesis.

A simpler proxy: compute the position of German verbs relative to English. If the German verb appears much earlier in the hypothesis than in the reference, this likely indicates a premature commitment.

**Deliverables for Phase 6:**
- Ablation tables for uncertainty signals, K values, wait-k interaction
- Error analysis breakdown by category (negation, numerals, entities, reordering)
- Before/after examples showing where selective refinement fixed specific errors

---

## 4. Team Division of Labor (Detailed)

| Week | Jeng-Yue | Wilson | Haoling |
|------|----------|--------|---------|
| 1 | Data pipeline, WMT download and prep, SimulEval format | Environment setup, model selection and testing | Literature review for uncertainty metrics, project setup docs |
| 2 | Verify baseline metrics match published numbers | Implement wait-k SimulEval agent | Set up plotting and analysis scripts |
| 3 | Implement compute-matched larger-beam baseline | Multi-candidate generation (Method A) | Begin designing uncertainty module API |
| 4 | Run Method A sweeps (K values) | Re-ranking implementations (log-prob, MBR) | Implement token entropy and log-prob computation |
| 5 | Prepare dev set for correlation analysis | Integrate uncertainty into agent | Run correlation analysis, produce diagnostic plots |
| 6 | Pre-tag dev set errors by category (negation, numerals, etc.) | Implement STTR agent (Method C) with gating | Compare uncertainty signals, select best |
| 7 | Run always-refine baseline (Method B) | τ sweep experiments, trigger rate analysis | Run wait-k × STTR interaction ablations |
| 8 | Error analysis by category | Results compilation, Pareto frontier plots | Draft ablation section of report |

---

## 5. Risk Mitigation Strategies

### Risk 1: Uncertainty Doesn't Correlate with Errors

**Likelihood:** Moderate (this is the main technical risk)
**Impact:** High — the gating mechanism becomes useless
**Mitigation:**
- Test multiple uncertainty signals (entropy, log-prob, MC dropout)
- Try different aggregation strategies (mean, max, top-3)
- Fall back to a fixed-schedule refinement (e.g., refine every N-th commit point) as a simpler baseline that still shows the value of second-pass refinement
- Even if the gate is imperfect, show that the system still improves over baseline by doing some refinement

### Risk 2: Refinement Doesn't Improve Quality

**Likelihood:** Low-moderate
**Impact:** High — Methods A, B, C all become pointless
**Mitigation:**
- Verify offline first: take baseline outputs, apply refinement to the full (non-streaming) translations, and check if quality improves. If it doesn't improve even offline, the refinement strategy itself is broken.
- Try a stronger refinement: use a larger model, more candidates, or LLM post-editing
- At minimum, "always refine" with K=16 candidates should beat K=1. If it doesn't, there's a bug.

### Risk 3: SimulEval Integration Issues

**Likelihood:** Moderate (SimulEval's agent interface has quirks)
**Impact:** Medium — delays timeline
**Mitigation:**
- Start with the dummy wait-k agent from SimulEval's examples to verify the evaluation pipeline works
- Use the text-to-text agent first (simpler than speech-to-text)
- SimulEval is no longer actively maintained — consider simulstream as a backup if SimulEval has compatibility issues with newer PyTorch/transformers versions
- Pin dependency versions in requirements.txt

### Risk 4: Compute Limitations

**Likelihood:** Depends on institution
**Impact:** Medium — can't run all ablations
**Mitigation:**
- WMT14 is only 3003 sentences — evaluation is fast even on CPU
- Use smaller MT models (opus-mt ~300M params) for development; switch to larger for final numbers
- Run development experiments on a random 500-sentence subset of WMT14
- Parallelize across team members: each person runs different ablation configs

### Risk 5: Model is "Confidently Wrong"

**Likelihood:** Moderate
**Impact:** Medium — the gate triggers on the wrong segments
**Mitigation:**
- Calibrate threshold on dev set (held-out from the set used to tune other hyperparameters)
- Report precision/recall of the uncertainty gate (what fraction of triggered refinements actually needed it?)
- Acknowledge this limitation transparently in the paper; it's a known issue in uncertainty estimation literature (cite Wang et al., 2020)

---

## 6. Repository Structure

```
sttr-project/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── method_a.yaml
│   ├── method_b.yaml
│   └── method_c.yaml
├── agents/
│   ├── waitk_agent.py           # Phase 2: baseline
│   ├── multicandidate_agent.py  # Phase 3: Method A
│   ├── always_refine_agent.py   # Phase 5: Method B
│   └── sttr_agent.py            # Phase 5: Method C
├── uncertainty/
│   ├── entropy.py               # Token and sequence entropy
│   ├── logprob.py               # Sequence log-probability
│   ├── mc_dropout.py            # MC Dropout variance
│   └── analysis.py              # Correlation analysis scripts
├── scripts/
│   ├── run_whisper_asr.py       # Phase 1: ASR pre-computation
│   ├── prepare_simuleval.py     # Phase 1: format data
│   ├── run_baseline.sh          # Phase 2: sweep wait-k
│   ├── run_method_a.sh          # Phase 3: sweep K
│   ├── run_sttr.sh              # Phase 5: sweep τ
│   └── run_ablations.sh         # Phase 6: all ablations
├── analysis/
│   ├── plot_pareto.py           # Quality-latency curves
│   ├── error_analysis.py        # Linguistically-driven error categorization
│   ├── trigger_rate.py          # Refinement trigger rate stats
│   └── correlation.py           # Uncertainty-error correlation
├── data/
│   ├── asr_outputs/             # Pre-computed Whisper transcriptions
│   └── simuleval_inputs/        # Formatted source/target files
├── outputs/                     # SimulEval output directories
└── report/
    ├── main.tex
    ├── figures/
    └── tables/
```

---

## 7. Key Milestones and Go/No-Go Checkpoints

| Milestone | Week | Go/No-Go Criteria |
|-----------|------|-------------------|
| **Baseline established** | 2 | BLEU within 3 points of published numbers |
| **Method A shows improvement** | 4 | Multi-candidate beats single-beam at matched compute |
| **Uncertainty correlates with errors** | 6 | Spearman ρ > 0.15 between uncertainty and error rate |
| **STTR improves Pareto frontier** | 7 | At least one τ setting beats baseline AND always-refine on the BLEU-LAAL curve |
| **Final results compiled** | 8 | Complete tables, plots, error analysis ready for report |

If the Phase 6 go/no-go fails (uncertainty doesn't correlate), pivot to:
- Fixed-schedule refinement (refine every N-th step) as a simpler alternative
- Oracle gating (use reference BLEU to decide when to refine) to show the potential ceiling
- Focus the paper on the analysis of *why* uncertainty estimation fails in streaming MT, which is itself a valid research contribution

---

## 8. Expected Outputs for Final Report

1. **Table 1:** Main results comparing Baseline, Method A, Method B (always-refine), Method C (STTR) across BLEU, COMET, AL, LAAL
2. **Figure 1:** BLEU vs. LAAL Pareto frontier with all methods overlaid
3. **Figure 2:** Trigger rate vs. τ, with BLEU overlaid
4. **Table 2:** Uncertainty signal comparison (entropy vs. log-prob vs. MC dropout)
5. **Table 3:** Error analysis by category (negation, numerals, entities, reordering)
6. **Figure 3:** Example translations showing where selective refinement fixed errors
7. **Table 4:** Wait-k × STTR interaction ablation
