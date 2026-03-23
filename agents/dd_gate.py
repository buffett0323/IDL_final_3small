"""Distribution Divergence (DD) gate for streaming EN->ZH translation.

Computes avg JS divergence across K future-conditioned Chinese next-token
distributions.  Reuses the agent's already-loaded base model — no extra
model loading needed.

Key design:
  - futures (oracle mode): sample K English "futures" from the FULL source
    sentence by revealing 1..K additional source words beyond prefix_len
    (truncation mode, deterministic).  Requires oracle_source_words.
  - futures (lm_sample mode): use a separate English LM to generate K
    diverse continuations of the current prefix — no oracle needed.
    This is the realistic inference-time mode.
  - distributions: for each future, run the base MT model to obtain the next
    Chinese token distribution (first N steps).
  - DD score: average pairwise JS divergence across the K distributions,
    averaged over the first N decoding steps (avg_js_firstN).
  - gate: if DD score <= tau -> COMMIT; else -> READ.

Multi-step support:
  - seq2seq (NLLB):  up to n_steps free Chinese token distributions (skipping
    the forced-BOS step). K futures are batched into a single generate() call.
  - causal LM (Qwen4B-Base): autoregressive single-pass, n_steps distributions
    starting from the "Chinese:" boundary. K futures batched via left-padding.

Oracle source requirement (oracle mode only):
  - In online streaming, the agent only sees words 0..prefix_len-1. To get
    genuine future diversity, we pass oracle_source_words (the full sentence,
    loaded from the source file at startup). This gives an upper bound on what
    a perfect English future-sampling LM could achieve, and is the correct
    first experiment before implementing a real LM-based future sampler.

LM-sample mode:
  - future_lm / future_lm_tokenizer: a separate causal LM used ONLY to
    generate English future continuations (not to translate).
  - Recommended: Qwen3-4B-Base or similar (small, fast, decent English).
  - Temperature sampling with T=0.9 provides diverse futures.
  - Division of labour:
      future LM  -> generates K English futures (plausible continuations)
      MT model   -> computes Chinese next-token distributions per future
      DD gate    -> compares distributions, decides READ/COMMIT

Policy score hierarchy:
  - avg_js_first1 : JS at step 1 only (fastest proxy).
  - avg_js_first3 : avg JS over first 3 steps (balanced default).
  - avg_js_first5 : avg JS over first 5 steps (smoother, requires n_steps>=5).
  - avg_js_firstN : avg JS over ALL computed steps (= avg_js_first{n_steps}).
  The gate uses avg_js_firstN where N = n_steps (configured via --dd-steps).

This module is intentionally minimal and does NOT import from
future_consistency.py to avoid circular dependencies or heavy imports.
"""
from __future__ import annotations

import re
import math
from typing import Optional

import torch
import torch.nn.functional as F


# ── JS utilities ──────────────────────────────────────────────────────────────

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """Symmetric Jensen-Shannon divergence (log-space, numerically stable).

    JS(P‖Q) = 0.5 * KL(P‖M) + 0.5 * KL(Q‖M),  M = 0.5*(P+Q)
    Returns a non-negative float (0 = identical distributions).
    """
    p = p.float().clamp(min=eps)
    q = q.float().clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum().item()
    kl_qm = (q * (q / m).log()).sum().item()
    return max(0.0, 0.5 * (kl_pm + kl_qm))


@torch.no_grad()
def sample_lm_futures(
    prefix_text: str,
    lm_model,
    lm_tokenizer,
    device: str,
    K: int = 4,
    future_words: int = 15,
    temperature: float = 0.9,
) -> list[str]:
    """Use a causal LM to sample K plausible English continuations.

    The model is prompted to complete a news sentence.  Each of the K returned
    strings is the FULL source (prefix + sampled continuation), ready to be fed
    to the MT model for distribution computation.

    Args:
        prefix_text : observed English source so far (e.g. "Ana Balarin, partner and ECD")
        lm_model    : causal LM (e.g. Qwen3-4B-Base) — NOT the MT model.
        lm_tokenizer: corresponding tokenizer.
        device      : PyTorch device string.
        K           : number of futures to sample.
        future_words: max new tokens to generate per future.
        temperature : sampling temperature; higher → more diverse futures.

    Returns:
        list[K] of strings, each = prefix_text + generated_continuation.
    """
    # Prompt: ask model to continue a news sentence.
    # Using a short instruction works well for both base and instruct causal LMs.
    prompt = (
        "Continue the following news sentence naturally with about "
        f"{future_words} more words:\n\n"
        f'"{prefix_text}'
    )

    inputs = lm_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Build stop token list: EOS + common instruct-model turn separators
    eos_id = lm_tokenizer.eos_token_id or 0
    stop_ids = [eos_id]
    for tok in ["<|im_end|>", "<|endoftext|>", "\n"]:
        tid = lm_tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != lm_tokenizer.unk_token_id:
            stop_ids.append(tid)
    stop_ids = list(set(stop_ids))

    outputs = lm_model.generate(
        **inputs,
        max_new_tokens=future_words,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=K,
        pad_token_id=lm_tokenizer.pad_token_id or eos_id,
        eos_token_id=stop_ids,
        repetition_penalty=1.1,
    )

    futures: list[str] = []
    seen: set[str] = set()
    for seq in outputs:
        continuation = lm_tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        # Keep only the first line / sentence fragment (stop at newline or closing quote)
        continuation = re.split(r'[\n"\u201d]', continuation)[0].strip()
        # Remove any trailing punctuation that ends a sentence
        if continuation and continuation[-1] in ".!?":
            continuation = continuation[:-1]
        full = (prefix_text + " " + continuation).strip()
        if full not in seen:
            seen.add(full)
            futures.append(full)

    # Deduplicate: if we got fewer unique futures, pad with prefix_text itself
    while len(futures) < K:
        futures.append(prefix_text)

    return futures[:K]


def sample_truncation_futures(
    oracle_source_words: list[str], prefix_len: int, K: int
) -> list[str]:
    """Sample K futures by revealing 1..K more source words after prefix_len.

    IMPORTANT: oracle_source_words must be the FULL sentence (not just the
    observed prefix).  In a streaming agent, pass the full source loaded from
    the source file at startup — do NOT pass self.states.source (that is only
    the observed prefix and makes all futures identical, giving JS=0).

    Deterministic and fast.  All futures are nested prefixes of one another
    (tests same-path prefix-extension stability).
    """
    total = len(oracle_source_words)
    futures = []
    for k in range(1, K + 1):
        end = min(prefix_len + k, total)
        futures.append(" ".join(oracle_source_words[:end]))
        if end >= total:
            break
    # Pad remaining slots with the full sentence
    full = " ".join(oracle_source_words)
    while len(futures) < K:
        futures.append(full)
    return futures[:K]


# ── Batched distribution getters ──────────────────────────────────────────────

@torch.no_grad()
def _get_dists_seq2seq_batched(
    model,
    tokenizer,
    forced_bos_token_id: Optional[int],
    source_texts: list[str],
    device: str,
    n_steps: int,
) -> list[list[torch.Tensor]]:
    """Get first n_steps Chinese token distributions for K seq2seq futures.

    Batches all K futures into a SINGLE model.generate() call for efficiency
    (replaces K sequential calls, giving ~K× speedup).

    For NLLB, step 0 of generate() is the forced-BOS (zho_Hans) which is
    identical across all futures -> skip it.  Useful distributions start at
    step 1.

    Returns: list[K] of list[n_steps] of CPU float tensors, shape (vocab,).
    """
    K = len(source_texts)
    enc = tokenizer(
        source_texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    gen_kwargs: dict = dict(
        max_new_tokens=n_steps + 1,  # +1 to account for the forced-BOS step
        output_scores=True,
        return_dict_in_generate=True,
        num_beams=1,
    )
    if forced_bos_token_id is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

    out = model.generate(**enc, **gen_kwargs)

    # out.scores[i] has shape [K, vocab_size] when batch_size=K, num_beams=1
    skip = 1 if forced_bos_token_id is not None and len(out.scores) > 1 else 0
    result: list[list[torch.Tensor]] = [[] for _ in range(K)]
    for idx in range(skip, min(skip + n_steps, len(out.scores))):
        probs_batch = F.softmax(out.scores[idx].float(), dim=-1).cpu()  # [K, vocab]
        for k in range(K):
            result[k].append(probs_batch[k])
    return result


@torch.no_grad()
def _get_dists_causal_batched(
    model,
    tokenizer,
    prompt_template: str,
    source_texts: list[str],
    device: str,
    n_steps: int,
) -> list[list[torch.Tensor]]:
    """Get first n_steps Chinese token distributions for K causal-LM futures.

    Batches all K prompts using LEFT padding (required for causal LM batching
    so that all sequences end at the same position) and runs n_steps greedy
    autoregressive steps.

    Returns: list[K] of list[n_steps] of CPU float tensors, shape (vocab,).
    """
    K = len(source_texts)
    prompts = [prompt_template.format(source=src) for src in source_texts]

    # Left-pad for causal LM so all sequences share the same "next" position
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    tokenizer.padding_side = orig_padding_side

    cur_ids = enc.input_ids.to(device)
    cur_mask = enc.attention_mask.to(device)

    result: list[list[torch.Tensor]] = [[] for _ in range(K)]
    for _ in range(n_steps):
        out = model(input_ids=cur_ids, attention_mask=cur_mask)
        logits = out.logits[:, -1, :].float()   # [K, vocab]
        probs_batch = F.softmax(logits, dim=-1).cpu()
        for k in range(K):
            result[k].append(probs_batch[k])
        # Greedy next token for all K sequences (reproducible, minimal variance)
        next_toks = logits.argmax(dim=-1).unsqueeze(1)  # [K, 1]
        cur_ids = torch.cat([cur_ids, next_toks], dim=1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones(K, 1, device=device, dtype=cur_mask.dtype)],
            dim=1,
        )
    return result


# ── JS aggregation ────────────────────────────────────────────────────────────

def _avg_js_over_futures_and_steps(
    dists_per_future: list[list[torch.Tensor]],
) -> dict[str, float]:
    """Compute per-step and aggregated JS divergence across K futures.

    Returns:
        avg_js_first1 : pairwise avg JS at step 1 only (fast proxy).
        avg_js_first3 : avg JS over first 3 steps (balanced default).
        avg_js_first5 : avg JS over first 5 steps (smoother signal).
        avg_js_firstN : avg JS over ALL available steps (= firstN where N=n_steps).
        per_step_js   : list of per-step avg JS values.

    The POLICY score is avg_js_firstN (configurable via --dd-steps=N).
    avg_js_first1/3/5 are logged for analysis only.
    """
    K = len(dists_per_future)
    _zero = {"avg_js_first1": 0.0, "avg_js_first3": 0.0,
             "avg_js_first5": 0.0, "avg_js_firstN": 0.0, "per_step_js": []}
    if K < 2:
        return _zero

    n_steps = min(len(d) for d in dists_per_future)
    if n_steps == 0:
        return _zero

    per_step_js: list[float] = []
    for step_i in range(n_steps):
        step_dists = [dists_per_future[k][step_i] for k in range(K)]
        pairs = [
            js_divergence(step_dists[i], step_dists[j])
            for i in range(K) for j in range(i + 1, K)
        ]
        per_step_js.append(sum(pairs) / len(pairs))

    def _avg_over(steps: list[float], n: int) -> float:
        trunc = steps[:n]
        return sum(trunc) / len(trunc) if trunc else 0.0

    return {
        "avg_js_first1": per_step_js[0] if per_step_js else 0.0,
        "avg_js_first3": _avg_over(per_step_js, 3),
        "avg_js_first5": _avg_over(per_step_js, 5),
        "avg_js_firstN": sum(per_step_js) / len(per_step_js),
        "per_step_js": per_step_js,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_dd_score(
    model,
    tokenizer,
    oracle_source_words: list[str],
    prefix_len: int,
    device: str,
    *,
    causal_lm: bool,
    forced_bos_token_id: Optional[int] = None,
    prompt_template: Optional[str] = None,
    K: int = 4,
    n_steps: int = 3,
    # ── LM-sample futures ──────────────────────────────────────────────────
    future_mode: str = "oracle",        # "oracle" | "lm_sample"
    future_lm=None,                     # causal LM for English future generation
    future_lm_tokenizer=None,           # tokenizer for future_lm
    future_words: int = 15,             # tokens to generate per future
    future_temperature: float = 0.9,    # sampling temperature for diversity
) -> dict[str, float]:
    """Compute the DD score for the current source prefix.

    Two future-sampling modes:

    oracle (default):
        Uses the FULL oracle source sentence to deterministically reveal
        1..K more words.  Requires oracle_source_words (not truncated).
        This is the upper-bound experiment.

    lm_sample:
        Uses future_lm to generate K diverse English continuations of the
        current prefix — NO oracle required.  This is the realistic mode
        for actual deployment.  future_lm / future_lm_tokenizer must be set.

    Policy gate scalar: avg_js_firstN  (average JS over first n_steps).
    Also returns avg_js_first1 / avg_js_first3 / avg_js_first5 for logging.

    Args:
        model / tokenizer   : already-loaded base MT model (NLLB or Qwen).
        oracle_source_words : FULL source sentence split into words (oracle mode).
                              Ignored in lm_sample mode.
        prefix_len          : number of source words currently observed.
        device              : PyTorch device string.
        causal_lm           : True for decoder-only MT model (Qwen), False for NLLB.
        forced_bos_token_id : NLLB target language BOS id (None for causal).
        prompt_template     : few-shot prompt string with {source} placeholder.
        K                   : number of futures to sample.
        n_steps             : number of MT decoding steps to average JS over.
        future_mode         : "oracle" (default) or "lm_sample".
        future_lm           : causal LM for future generation (lm_sample mode).
        future_lm_tokenizer : tokenizer for future_lm.
        future_words        : max tokens to generate per sampled future.
        future_temperature  : sampling temperature (higher = more diverse).

    Returns dict with keys:
        avg_js_first1, avg_js_first3, avg_js_first5, avg_js_firstN,
        per_step_js, K, n_steps, futures, future_mode.
    """
    if future_mode == "lm_sample":
        assert future_lm is not None and future_lm_tokenizer is not None, (
            "future_lm and future_lm_tokenizer must be set when future_mode='lm_sample'"
        )
        prefix_text = " ".join(oracle_source_words[:prefix_len])
        futures = sample_lm_futures(
            prefix_text,
            future_lm,
            future_lm_tokenizer,
            device,
            K=K,
            future_words=future_words,
            temperature=future_temperature,
        )
    else:
        futures = sample_truncation_futures(oracle_source_words, prefix_len, K)

    if causal_lm:
        assert prompt_template is not None, "prompt_template required for causal LM"
        dists = _get_dists_causal_batched(
            model, tokenizer, prompt_template, futures, device, n_steps
        )
    else:
        dists = _get_dists_seq2seq_batched(
            model, tokenizer, forced_bos_token_id, futures, device, n_steps
        )

    stats = _avg_js_over_futures_and_steps(dists)
    stats["K"] = K
    stats["n_steps"] = n_steps
    stats["futures"] = futures
    stats["future_mode"] = future_mode
    return stats
