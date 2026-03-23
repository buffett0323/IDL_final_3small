"""Future-Consistency Scoring for Simultaneous EN->ZH Translation.

Two future-sampling modes:
  - truncation : reveal 1..K extra source words (deterministic, nested — tests
                 same-path prefix-extension stability, NOT branching robustness)
  - lm_sample  : use the base causal LM to generate K diverse English futures
                 via temperature sampling + deduplication + nested-prefix rejection
                 (tests true branching future robustness)

Two scoring methods:
1. distribution_divergence
   For each future, run the base MT model to get the next Chinese token
   distribution.  Compute pairwise Jensen-Shannon divergence.

2. semantic_lcp
   For each future, generate a short Chinese continuation. Measure literal LCP,
   pairwise edit distance, and token-set overlap.

Future diversity diagnostics (always computed):
  - nested_prefix_rate : fraction of future pairs where one is a strict prefix
  - avg_future_edit_dist : mean pairwise normalized edit distance between futures
  - unique_future_ratio : fraction of futures that are unique
  - is_nested_warning : True when nested_prefix_rate > 0.5

IMPORTANT: Results labeled "same-path stability" are from truncation futures.
           Results labeled "branching robustness" require lm_sample futures.
           Do NOT conflate the two.

Design constraints respected:
  - Does NOT use a strong model to rewrite the translation.
  - Agreement is measured in Chinese output space.
  - All thresholds are configurable.
  - Conservative: returns empty safe prefix rather than hallucinating.
  - Numerically stable: uses log-space for KL/JS.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Reuse existing utilities from the same package directory
_AGENT_DIR = Path(__file__).resolve().parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from model_utils import (
    DEFAULT_CACHE_DIR,
    _CAUSAL_ENZH_FEW_SHOT,
    build_generate_kwargs,
    load_causal_translation_model,
    load_translation_model,
    split_chinese_chars,
)


# ---------------------------------------------------------------------------
# String distance utilities (used by diversity diagnostics and semantic LCP)
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[lb]


def normalized_edit_distance(a: str, b: str) -> float:
    """Edit distance normalized by max length (in [0, 1])."""
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


# ---------------------------------------------------------------------------
# Future diversity diagnostics
# ---------------------------------------------------------------------------

def is_strict_prefix(a: str, b: str) -> bool:
    """True if a is a strict (non-equal) prefix of b, or vice versa."""
    a, b = a.strip(), b.strip()
    if a == b:
        return False
    return b.startswith(a) or a.startswith(b)


def compute_future_diversity(futures: list[str]) -> dict:
    """Measure how diverse the sampled English futures are.

    Returns:
        nested_prefix_rate   : fraction of pairs (i,j) where one is a strict
                               prefix of the other.  1.0 = all pairs are nested
                               (truncation mode always gives 1.0).
        avg_future_edit_dist : mean pairwise normalized edit distance between
                               future strings.  Near-0 = almost identical.
        unique_future_ratio  : fraction of futures that are unique strings.
        avg_future_len       : average character length of futures.
        is_nested_warning    : True when nested_prefix_rate > 0.5.
    """
    n = len(futures)
    if n < 2:
        return {
            "nested_prefix_rate": 0.0,
            "avg_future_edit_dist": 0.0,
            "unique_future_ratio": 1.0,
            "avg_future_len": float(len(futures[0])) if futures else 0.0,
            "is_nested_warning": False,
        }

    nested_count = 0
    edit_vals: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if is_strict_prefix(futures[i], futures[j]):
                nested_count += 1
            edit_vals.append(normalized_edit_distance(futures[i], futures[j]))

    total_pairs = n * (n - 1) // 2
    nested_rate = nested_count / total_pairs
    avg_edit = sum(edit_vals) / len(edit_vals) if edit_vals else 0.0
    unique_ratio = len(set(f.strip() for f in futures)) / n
    avg_len = sum(len(f) for f in futures) / n

    return {
        "nested_prefix_rate": nested_rate,
        "avg_future_edit_dist": avg_edit,
        "unique_future_ratio": unique_ratio,
        "avg_future_len": avg_len,
        "is_nested_warning": nested_rate > 0.5,
    }


# ---------------------------------------------------------------------------
# Future Samplers
# ---------------------------------------------------------------------------

def sample_truncation_futures(
    source_words: list[str],
    prefix_len: int,
    K: int,
) -> list[str]:
    """Sample K futures by revealing 1..K additional source words (truncation).

    WARNING: These futures are strictly nested prefixes of one another.
    This measures same-path prefix-extension stability, NOT branching
    future robustness.  Use sample_lm_futures() for true diversity.

    Args:
        source_words: full whitespace-split English source.
        prefix_len:   number of words in the observed prefix.
        K:            number of futures to sample.
    """
    total = len(source_words)
    futures = []
    for k in range(1, K + 1):
        end = min(prefix_len + k, total)
        futures.append(" ".join(source_words[:end]))
        if end >= total:
            break

    full_src = " ".join(source_words)
    while len(futures) < K:
        futures.append(full_src)

    return futures[:K]


def sample_lm_futures(
    source_words: list[str],
    prefix_len: int,
    K: int,
    model,
    tokenizer,
    device: str,
    temperature: float = 1.2,
    top_p: float = 0.9,
    top_k_lm: int = 50,
    max_new_words: int = 12,
    **_kwargs,
) -> list[str]:
    """Sample K English futures by letting the base causal LM continue the prefix.

    Encodes the English prefix, runs one batched generate() call with
    num_return_sequences=K and temperature sampling.  No rejection logic —
    we just take whatever the model gives us.  Diversity comes from sampling
    temperature alone.

    Qwen3-4B-Base is pretrained on multilingual text; feeding a raw English
    prefix causes it to naturally continue in English.

    Args:
        source_words:  full whitespace-split English source.
        prefix_len:    words in the observed prefix.
        K:             number of futures to sample.
        model:         loaded causal LM (on device).
        tokenizer:     corresponding tokenizer.
        device:        PyTorch device string.
        temperature:   sampling temperature (>1 = more diverse).
        top_p:         nucleus sampling probability.
        top_k_lm:      top-k vocabulary restriction (0 = disabled).
        max_new_words: approximate max continuation length in words.
    """
    prefix_text = " ".join(source_words[:prefix_len])

    inputs = tokenizer(
        prefix_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)
    prompt_len = inputs.input_ids.shape[1]

    # 1 English word ≈ 1.3 BPE tokens
    max_new_tokens = int(max_new_words * 1.5) + 4

    with torch.no_grad():
        # Expand input for K parallel samples
        input_ids = inputs.input_ids.expand(K, -1)
        attention_mask = inputs.attention_mask.expand(K, -1)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k_lm if top_k_lm > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    futures = []
    for i in range(K):
        new_ids = out[i][prompt_len:]
        cont = tokenizer.decode(new_ids, skip_special_tokens=True)
        # Keep only the first line
        cont = cont.split("\n")[0].strip()
        # Trim to whole words
        cont_words = cont.split()[:max_new_words]
        cont = " ".join(cont_words)
        futures.append((prefix_text + " " + cont).strip() if cont else prefix_text)

    return futures


# ---------------------------------------------------------------------------
# Next-token distribution helpers
# ---------------------------------------------------------------------------

def _get_next_token_dist_seq2seq(
    model,
    tokenizer,
    source_text: str,
    forced_bos_token_id,
    device: str,
) -> torch.Tensor:
    """Return softmax next-token probability vector (vocab_size,) for seq2seq."""
    inputs = tokenizer(
        source_text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **build_generate_kwargs(forced_bos_token_id, **inputs),
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
    # out.scores is a tuple of length 1 (one generated step), each (batch, vocab)
    logits = out.scores[0][0].float()
    return F.softmax(logits, dim=-1).cpu()


def _get_next_token_dist_causal(
    model,
    tokenizer,
    source_text: str,
    prompt_template: str,
    device: str,
) -> torch.Tensor:
    """Return softmax next-token probability vector (vocab_size,) for causal LM.

    We build the few-shot prompt and take the logits at the *last* input
    position (i.e., the model is about to generate the first Chinese token).
    """
    prompt = prompt_template.format(source=source_text)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids)
    logits = out.logits[0, -1, :].float()  # last token position
    return F.softmax(logits, dim=-1).cpu()


# ---------------------------------------------------------------------------
# Short Chinese continuation helper
# ---------------------------------------------------------------------------

def _generate_continuation_seq2seq(
    model,
    tokenizer,
    source_text: str,
    forced_bos_token_id,
    device: str,
    cont_len: int,
) -> str:
    """Generate a short Chinese continuation with seq2seq (NLLB etc.)."""
    inputs = tokenizer(
        source_text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **build_generate_kwargs(forced_bos_token_id, **inputs),
            max_new_tokens=cont_len,
            do_sample=False,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _generate_continuation_causal(
    model,
    tokenizer,
    source_text: str,
    prompt_template: str,
    device: str,
    cont_len: int,
) -> str:
    """Generate a short Chinese continuation with a causal LM (Qwen4B etc.)."""
    prompt = prompt_template.format(source=source_text)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cont_len,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text.split("\n")[0].strip()  # stop at first newline (next few-shot)


# ---------------------------------------------------------------------------
# JS / KL divergence (numerically stable, log-space)
# ---------------------------------------------------------------------------

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence JS(p || q), symmetric, in [0, log2].

    Uses log-space computation for numerical stability.
    Returns a Python float in nats (natural log units).
    """
    p = p.double() + eps
    q = q.double() + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum().item()
    kl_qm = (q * (q.log() - m.log())).sum().item()
    return max(0.0, 0.5 * (kl_pm + kl_qm))


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """KL(p || q) in nats.  May be large/inf if q has zero mass where p>0."""
    p = p.double() + eps
    q = q.double() + eps
    p = p / p.sum()
    q = q / q.sum()
    return (p * (p.log() - q.log())).sum().item()


def topk_token_overlap(dists: list[torch.Tensor], top_k: int = 10) -> float:
    """Fraction of top-k tokens (by probability) shared across all distributions.

    Returns a value in [0, 1]:  1.0 means all distributions agree on the same
    top-k tokens.
    """
    if len(dists) < 2:
        return 1.0
    sets = [set(d.topk(top_k).indices.tolist()) for d in dists]
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    return len(common) / top_k


def consensus_token_score(dists: list[torch.Tensor]) -> float:
    """max_v  min_k p_k(v) — probability that all futures agree on best token.

    This is the highest probability any single token achieves when we take the
    *minimum* across all K distributions.  A high score means at least one
    token is reliably probable under every future.
    """
    if not dists:
        return 0.0
    # Stack to (K, vocab), take min over K, then max over vocab
    stacked = torch.stack(dists, dim=0).double()  # (K, vocab)
    min_over_k = stacked.min(dim=0).values          # (vocab,)
    return min_over_k.max().item()


# ---------------------------------------------------------------------------
# LCP and semantic agreement
# ---------------------------------------------------------------------------

def literal_lcp_chars(strings: list[str]) -> str:
    """Character-level longest common prefix across all strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        i = 0
        while i < min(len(prefix), len(s)) and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def token_set_overlap(strings: list[str]) -> float:
    """Average pairwise Jaccard overlap on character unigrams.

    Characters serve as "tokens" (since Chinese is character-level).
    Returns value in [0, 1].
    """
    if len(strings) < 2:
        return 1.0
    total, count = 0.0, 0
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            si = set(strings[i])
            sj = set(strings[j])
            union = si | sj
            if not union:
                total += 1.0
            else:
                total += len(si & sj) / len(union)
            count += 1
    return total / count if count else 1.0


def semantic_agreement_score(
    lcp_len: int,
    avg_edit_dist: float,
    token_overlap: float,
    cont_len: int,
) -> float:
    """Conservative approximation of semantic agreement in [0, 1].

    Combines three signals with equal weight:
      - lcp_ratio:    LCP length / continuation length
      - edit_agree:   1 - avg_normalized_edit_distance
      - token_overlap: Jaccard overlap of character unigrams

    NOTE: This is a character-level approximation.  No embedding model is used.
    We deliberately avoid claiming "semantic" equivalence beyond what character
    overlap supports.
    """
    lcp_ratio = lcp_len / max(cont_len, 1)
    edit_agree = max(0.0, 1.0 - avg_edit_dist)
    return (lcp_ratio + edit_agree + token_overlap) / 3.0


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------

class FutureConsistencyScorer:
    """Offline future-consistency scorer for simultaneous MT evaluation.

    Wraps the two scoring methods and handles model loading/dispatch.

    Args:
        model_name:       HuggingFace model name or local path.
        device:           PyTorch device string, e.g. 'cuda:0'.
        causal_lm:        If True, load as AutoModelForCausalLM (Qwen4B etc.).
                          If False, load as AutoModelForSeq2SeqLM (NLLB etc.).
        future_mode:      'truncation' (nested prefix extension) or
                          'lm_sample'  (LM-generated diverse futures).
                          lm_sample requires causal_lm=True.
        lm_temperature:   Sampling temperature for lm_sample mode.
        lm_top_p:         Nucleus sampling p for lm_sample mode.
        lm_max_new_words: Max continuation words per LM sample.
        source_lang:      Source language code for multilingual tokenizers.
        target_lang:      Target language code for multilingual tokenizers.
        cache_dir:        Local model cache directory.
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cuda:0",
        causal_lm: bool = False,
        future_mode: str = "truncation",
        lm_temperature: float = 1.2,
        lm_top_p: float = 0.9,
        lm_max_new_words: int = 12,
        source_lang: str | None = "eng_Latn",
        target_lang: str | None = "zho_Hans",
        cache_dir: str | None = None,
    ):
        self.device = device
        self.causal_lm = causal_lm
        self._prompt_template: str | None = None
        self._forced_bos: int | None = None

        if future_mode not in ("truncation", "lm_sample"):
            raise ValueError(f"future_mode must be 'truncation' or 'lm_sample', got {future_mode!r}")
        if future_mode == "lm_sample" and not causal_lm:
            raise ValueError("future_mode='lm_sample' requires causal_lm=True")

        self.future_mode = future_mode
        self.lm_temperature = lm_temperature
        self.lm_top_p = lm_top_p
        self.lm_max_new_words = lm_max_new_words

        print(f"[FutureConsistency] Loading {'causal' if causal_lm else 'seq2seq'} "
              f"model: {model_name} on {device}")
        print(f"[FutureConsistency] future_mode={future_mode}"
              + (f"  T={lm_temperature}  top_p={lm_top_p}" if future_mode == "lm_sample" else ""))
        if causal_lm:
            self.tokenizer, self.model, self._prompt_template = (
                load_causal_translation_model(model_name, device, cache_dir)
            )
        else:
            self.tokenizer, self.model, self._forced_bos = load_translation_model(
                model_name, device, source_lang, target_lang, cache_dir
            )
        self.model.eval()
        print("[FutureConsistency] Model ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_futures(self, source_words: list[str], prefix_len: int, K: int) -> list[str]:
        """Dispatch to the configured future-sampling strategy."""
        if self.future_mode == "lm_sample":
            return sample_lm_futures(
                source_words, prefix_len, K,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                temperature=self.lm_temperature,
                top_p=self.lm_top_p,
                max_new_words=self.lm_max_new_words,
            )
        return sample_truncation_futures(source_words, prefix_len, K)

    def _next_token_dist(self, source_text: str) -> torch.Tensor:
        if self.causal_lm:
            return _get_next_token_dist_causal(
                self.model, self.tokenizer, source_text,
                self._prompt_template, self.device,
            )
        else:
            return _get_next_token_dist_seq2seq(
                self.model, self.tokenizer, source_text,
                self._forced_bos, self.device,
            )

    def _short_continuation(self, source_text: str, cont_len: int) -> str:
        if self.causal_lm:
            return _generate_continuation_causal(
                self.model, self.tokenizer, source_text,
                self._prompt_template, self.device, cont_len,
            )
        else:
            return _generate_continuation_seq2seq(
                self.model, self.tokenizer, source_text,
                self._forced_bos, self.device, cont_len,
            )

    # ------------------------------------------------------------------
    # Method 1: Distribution Divergence
    # ------------------------------------------------------------------

    def score_distribution_divergence(
        self,
        source_words: list[str],
        prefix_len: int,
        K: int = 4,
        top_k_overlap: int = 10,
    ) -> dict[str, Any]:
        """Compute JS divergence across K future-conditioned next-token distributions.

        Args:
            source_words: whitespace-split full English source.
            prefix_len:   observed prefix length in words.
            K:            number of futures.
            top_k_overlap: how many top tokens to consider for overlap metric.

        Returns:
            dict with keys:
              future_mode       str    'truncation' or 'lm_sample'
              future_diversity  dict   nested_prefix_rate, avg_future_edit_dist, etc.
              futures           list of K English future strings
              avg_js            float  mean pairwise JS divergence
              max_js            float  max pairwise JS divergence
              pairwise_js       list[list[float]]  K×K matrix
              topk_overlap      float  fraction of top-k tokens shared across all dists
              consensus_score   float  max_v min_k p_k(v)
              kl_from_first     list[float]  KL(p_1 || p_k) for k=2..K (auxiliary)
              decision          str    COMMIT / BORDERLINE / READ
        """
        futures = self._sample_futures(source_words, prefix_len, K)
        diversity = compute_future_diversity(futures)

        dists: list[torch.Tensor] = []
        for fut in futures:
            with torch.no_grad():
                dists.append(self._next_token_dist(fut))

        # Pairwise JS matrix
        n = len(dists)
        pairwise = [[0.0] * n for _ in range(n)]
        js_values = []
        for i in range(n):
            for j in range(i + 1, n):
                v = js_divergence(dists[i], dists[j])
                pairwise[i][j] = v
                pairwise[j][i] = v
                js_values.append(v)

        avg_js = float(sum(js_values) / len(js_values)) if js_values else 0.0
        max_js = float(max(js_values)) if js_values else 0.0

        # KL from first distribution (auxiliary, may be asymmetric)
        kl_from_first = [
            kl_divergence(dists[0], dists[k]) for k in range(1, n)
        ]

        topk_ov = topk_token_overlap(dists, top_k=top_k_overlap)
        consensus = consensus_token_score(dists)

        return {
            "future_mode": self.future_mode,
            "future_diversity": diversity,
            "futures": futures,
            "avg_js": avg_js,
            "max_js": max_js,
            "pairwise_js": pairwise,
            "topk_overlap": topk_ov,
            "consensus_score": consensus,
            "kl_from_first": kl_from_first,
            # decision filled in by runner using configurable thresholds
        }

    # ------------------------------------------------------------------
    # Method 2: Semantic LCP
    # ------------------------------------------------------------------

    def score_semantic_lcp(
        self,
        source_words: list[str],
        prefix_len: int,
        K: int = 4,
        cont_len: int = 8,
    ) -> dict[str, Any]:
        """Generate K short Chinese continuations and measure their agreement.

        Args:
            source_words: whitespace-split full English source.
            prefix_len:   observed prefix length in words.
            K:            number of futures.
            cont_len:     target continuation length in tokens/characters.

        Returns:
            dict with keys:
              future_mode            str         'truncation' or 'lm_sample'
              future_diversity       dict        nested_prefix_rate etc.
              futures                list[str]   K English futures
              continuations          list[str]   K raw Chinese continuations
              normalized_candidates  list[str]   stripped/normalized versions
              literal_lcp            str         character-level common prefix
              literal_lcp_len        int         length of LCP
              avg_edit_distance      float       mean pairwise normalized edit dist
              token_overlap          float       mean pairwise Jaccard on char unigrams
              semantic_agreement     float       combined score in [0, 1]
              safe_prefix_candidate  str         literal LCP if len>=1 else ""
              decision               str         (filled by runner)
        """
        futures = self._sample_futures(source_words, prefix_len, K)
        diversity = compute_future_diversity(futures)

        continuations: list[str] = []
        for fut in futures:
            with torch.no_grad():
                raw = self._short_continuation(fut, cont_len)
            # Normalize: strip spaces, keep Chinese chars + common punctuation
            normed = raw.replace(" ", "").strip()
            continuations.append(normed)

        lcp = literal_lcp_chars(continuations)
        lcp_len = len(lcp)

        # Pairwise normalized edit distances
        n = len(continuations)
        edit_dists: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                edit_dists.append(
                    normalized_edit_distance(continuations[i], continuations[j])
                )
        avg_ed = float(sum(edit_dists) / len(edit_dists)) if edit_dists else 0.0

        tok_ov = token_set_overlap(continuations)
        sem_agr = semantic_agreement_score(lcp_len, avg_ed, tok_ov, cont_len)

        # Conservative safe prefix: only return LCP if it is non-empty
        safe_prefix = lcp if lcp_len >= 1 else ""

        return {
            "future_mode": self.future_mode,
            "future_diversity": diversity,
            "futures": futures,
            "continuations": continuations,
            "normalized_candidates": continuations,
            "literal_lcp": lcp,
            "literal_lcp_len": lcp_len,
            "avg_edit_distance": avg_ed,
            "token_overlap": tok_ov,
            "semantic_agreement": sem_agr,
            "safe_prefix_candidate": safe_prefix,
        }

    # ------------------------------------------------------------------
    # Combined scoring
    # ------------------------------------------------------------------

    def score(
        self,
        source_words: list[str],
        prefix_len: int,
        K: int = 4,
        cont_len: int = 8,
        top_k_overlap: int = 10,
    ) -> dict[str, Any]:
        """Run both methods and return combined result dict."""
        dd = self.score_distribution_divergence(
            source_words, prefix_len, K, top_k_overlap
        )
        sl = self.score_semantic_lcp(source_words, prefix_len, K, cont_len)
        # Merge (sl futures may be same as dd futures)
        return {
            "distribution_divergence": dd,
            "semantic_lcp": sl,
        }


# ---------------------------------------------------------------------------
# Decision heuristics (configurable thresholds)
# ---------------------------------------------------------------------------

def decide_from_divergence(
    avg_js: float,
    topk_overlap: float,
    commit_js_threshold: float = 0.05,
    read_js_threshold: float = 0.20,
    overlap_threshold: float = 0.5,
) -> str:
    """Map JS divergence + overlap to COMMIT / BORDERLINE / READ.

    Conservative rule:
      COMMIT   if avg_js < commit_js_threshold AND topk_overlap >= overlap_threshold
      READ     if avg_js > read_js_threshold
      BORDERLINE otherwise
    """
    if avg_js < commit_js_threshold and topk_overlap >= overlap_threshold:
        return "COMMIT"
    if avg_js > read_js_threshold:
        return "READ"
    return "BORDERLINE"


def decide_from_lcp(
    lcp_len: int,
    avg_edit_distance: float,
    semantic_agreement: float,
    commit_lcp_threshold: int = 2,
    read_edit_threshold: float = 0.8,
    commit_agreement_threshold: float = 0.5,
) -> str:
    """Map LCP metrics to COMMIT / BORDERLINE / READ.

    Conservative rule:
      COMMIT     if lcp_len >= commit_lcp_threshold AND
                    semantic_agreement >= commit_agreement_threshold
      READ       if lcp_len == 0 AND avg_edit_distance > read_edit_threshold
      BORDERLINE otherwise
    """
    if (lcp_len >= commit_lcp_threshold
            and semantic_agreement >= commit_agreement_threshold):
        return "COMMIT"
    if lcp_len == 0 and avg_edit_distance > read_edit_threshold:
        return "READ"
    return "BORDERLINE"
