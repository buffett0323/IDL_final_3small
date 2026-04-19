"""Core utilities for future-sampled hard token-intersection consensus decoding.

Method 3 idea (rewritten to match consensus_decoding.py design):
  1. Sample K plausible English futures from the observed source prefix.
  2. For each future, construct a full hypothetical source and query the
     translator for the *next-token* distribution via the vLLM completion
     endpoint, forcing the committed Chinese prefix in the assistant turn.
  3. Filter disallowed tokens BEFORE consensus (special, English, garbled).
  4. Take a hard intersection over token IDs in the filtered top-k sets.
  5. If the intersection is non-empty, pick the token with the highest average
     probability across futures and append it to the pending delta.
  6. Stop when the intersection becomes empty or a max-step cap is reached.
  7. Trim pending tokens to a clean boundary before committing.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOP_K = 10


def normalize_zh(text: str) -> str:
    """Normalize Chinese text: strip whitespace, normalize unicode."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", "", text).strip()


def clean_model_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.split("<|im_end|>")[0]
    text = text.split("<|endoftext|>")[0]
    return text.strip()


def _single_token_text(tokenizer: Any, tok_id: int) -> str:
    return tokenizer.decode(
        [tok_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _parse_token_id_string(raw: str) -> Optional[int]:
    text = str(raw or "").strip()
    match = re.fullmatch(r"token_id:(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _disallowed_token_reason(tokenizer: Any, tok_id: int) -> Optional[str]:
    """Check if a token is disallowed for consensus commit."""
    if tok_id is None:
        return "missing_token_id"
    if tok_id in set(getattr(tokenizer, "all_special_ids", []) or []):
        return "special_token_id"
    token_text = _single_token_text(tokenizer, tok_id)
    forbidden_fragments = [
        "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>",
    ]
    if any(frag in token_text for frag in forbidden_fragments):
        return "special_token_text"
    if re.search(r"[A-Za-z]", token_text):
        return "ascii_letters"
    if "\ufffd" in token_text:
        return "replacement_char"
    if any(ch in {"\u200d", "\ufe0f"} for ch in token_text):
        return "zero_width_or_variation_selector"
    if any(unicodedata.category(ch) in {"Cc", "Cs"} for ch in token_text):
        return "control_or_surrogate"
    return None


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _http_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e


# ── Prompt construction ───────────────────────────────────────────────────────

def build_translation_probe_prompt(
    tokenizer: Any,
    full_source: str,
    target_prefix: str,
) -> str:
    """Build a prompt that forces the model to continue from target_prefix.

    Uses apply_chat_template to construct a proper chat format, then appends
    the committed Chinese text into the assistant turn so the model MUST
    continue from that exact prefix.
    """
    if not str(target_prefix or "").strip():
        messages = [{
            "role": "user",
            "content": (
                "[TASK]\n"
                "Translate the [INPUT] text into Chinese.\n\n"
                f"[INPUT]\n{full_source}\n\n"
                "[IMPORTANT]\n"
                "Start the Chinese translation from the beginning and output "
                "only the next continuation token(s)."
            ),
        }]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False,
        )
        prompt += "<|im_start|>assistant\n"
        return prompt

    messages = [{
        "role": "user",
        "content": (
            "[TASK]\n"
            "Translate the [INPUT] text into Chinese.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            "[IMPORTANT]\n"
            "A partial Chinese translation is already committed at the start "
            "of the assistant reply. You must continue from that exact prefix "
            "and produce only the continuation."
        ),
    }]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False,
    )
    prompt += "<|im_start|>assistant\n"
    prompt += target_prefix
    return prompt


def build_final_completion_prompt(
    tokenizer: Any,
    full_source: str,
    committed_text: str,
) -> str:
    """Build a prompt for force-finishing: continue from committed_text."""
    if not str(committed_text or "").strip():
        messages = [{
            "role": "user",
            "content": (
                "[TASK]\n"
                "Translate the [INPUT] text into Chinese.\n\n"
                f"[INPUT]\n{full_source}\n\n"
                "[IMPORTANT]\n"
                "Output the complete Chinese translation only."
            ),
        }]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False,
        )
        prompt += "<|im_start|>assistant\n"
        return prompt

    messages = [{
        "role": "user",
        "content": (
            "[TASK]\n"
            "Translate the [INPUT] text into Chinese.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            "[IMPORTANT]\n"
            "A partial Chinese translation is already committed at the start "
            "of the assistant reply. Continue from that prefix and output "
            "only the remaining continuation."
        ),
    }]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False,
    )
    prompt += "<|im_start|>assistant\n"
    prompt += committed_text
    return prompt


# ── Distribution helpers ──────────────────────────────────────────────────────

def _parse_completion_top_logprobs(
    top_logprobs: Optional[List[Optional[Dict[str, float]]]],
    tokenizer: Any,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """Parse vLLM completion endpoint top_logprobs into {token_id: prob}."""
    if not top_logprobs:
        return {}, {"reason": "missing_top_logprobs"}
    step = top_logprobs[0]
    if not step:
        return {}, {"reason": "empty_top_logprobs"}

    id_distribution: Dict[int, float] = {}
    unknown_tokens: List[str] = []
    for raw_token, logprob in step.items():
        tok_id = _parse_token_id_string(raw_token)
        if tok_id is None and tokenizer is not None:
            # Fallback: when vLLM runs without --return-tokens-as-token-ids,
            # keys are text strings — encode them back to token IDs.
            encoded = tokenizer.encode(str(raw_token), add_special_tokens=False)
            if len(encoded) == 1:
                tok_id = encoded[0]
        if tok_id is None:
            unknown_tokens.append(str(raw_token))
            continue
        id_distribution[tok_id] = float(math.exp(float(logprob)))

    token_ids = list(id_distribution.keys())
    return id_distribution, {
        "reason": "ok" if id_distribution else "no_token_ids_in_top_logprobs",
        "topk_token_ids": token_ids,
        "topk_token_texts": [_single_token_text(tokenizer, tid) for tid in token_ids],
        "topk_true_probs": [round(id_distribution[tid], 6) for tid in token_ids],
        "unknown_top_logprob_tokens": unknown_tokens,
    }


def filter_distribution(
    tokenizer: Any,
    id_distribution: Dict[int, float],
) -> Dict[int, float]:
    """Remove disallowed tokens from a distribution BEFORE consensus."""
    return {
        tok_id: prob
        for tok_id, prob in id_distribution.items()
        if _disallowed_token_reason(tokenizer, tok_id) is None
    }


def topk_token_ids(dist: Dict[int, float], k: int = TOP_K) -> List[int]:
    return [tid for tid, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]]


# ── Consensus ─────────────────────────────────────────────────────────────────

def choose_consensus_token(
    distributions: List[Dict[int, float]],
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Hard intersection consensus over filtered distributions."""
    if not distributions:
        return None, {"reason": "no_distributions"}

    candidate_lists = [topk_token_ids(dist, TOP_K) for dist in distributions]

    intersection = set(candidate_lists[0])
    for clist in candidate_lists[1:]:
        intersection &= set(clist)

    if not intersection:
        return None, {
            "reason": "empty_intersection",
            "topk_lists": candidate_lists,
        }

    best_token = max(
        intersection,
        key=lambda tok: sum(d.get(tok, 0.0) for d in distributions) / len(distributions),
    )
    return best_token, {
        "reason": "ok",
        "intersection": sorted(intersection),
        "avg_prob": sum(d.get(best_token, 0.0) for d in distributions) / len(distributions),
    }


# ── Pending token buffer management ──────────────────────────────────────────

def has_suspicious_tail(text: str, last_token_text: str) -> bool:
    if not text:
        return False
    if "\ufffd" in last_token_text:
        return True
    if "\ufffd" in text[-4:]:
        return True
    last_char = text[-1]
    if last_char in {"\u200d", "\ufe0f"}:
        return True
    if unicodedata.category(last_char) in {"Mn", "Mc", "Me", "Cc", "Cs"}:
        return True
    return False


def sanitize_and_trim_pending(
    tokenizer: Any,
    committed_text: str,
    pending_token_ids: List[int],
) -> Tuple[List[int], str]:
    """Sanitize pending tokens and trim to a clean boundary.

    Returns (trimmed_token_ids, decoded_delta_text).
    """
    # 1. Remove disallowed tokens
    kept = [
        tid for tid in pending_token_ids
        if _disallowed_token_reason(tokenizer, tid) is None
    ]
    # 2. Trim suspicious tail
    while kept:
        decoded = tokenizer.decode(kept, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        full_text = committed_text + decoded
        last_tok_text = _single_token_text(tokenizer, kept[-1])
        if not has_suspicious_tail(full_text, last_tok_text):
            return kept, decoded
        kept.pop()
    return [], ""


# ── VLLMCompletionClient ─────────────────────────────────────────────────────

class VLLMCompletionClient:
    """Thin wrapper around vLLM's OpenAI-compatible /completions endpoint."""

    def __init__(self, api_base: str, model_name: str, timeout: float = 120.0):
        base = api_base.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.api_base = base
        self.model = model_name
        self.timeout = timeout

    def get_next_token_distribution(
        self,
        tokenizer: Any,
        full_source: str,
        target_prefix: str,
        top_k: int = TOP_K,
    ) -> Dict[int, float]:
        """Get filtered next-token distribution for a single prompt."""
        prompt = build_translation_probe_prompt(tokenizer, full_source, target_prefix)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": top_k,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        data = _http_json(
            f"{self.api_base}/completions",
            payload=payload,
            timeout=self.timeout,
        )
        choices = data.get("choices", [])
        if not choices:
            return {}
        choice = choices[0]
        logprobs = choice.get("logprobs", {}) if isinstance(choice, dict) else {}
        dist, _ = _parse_completion_top_logprobs(
            logprobs.get("top_logprobs"), tokenizer=tokenizer,
        )
        return filter_distribution(tokenizer, dist)

    def batch_get_next_token_distributions(
        self,
        tokenizer: Any,
        full_sources: List[str],
        target_prefix: str,
        top_k: int = TOP_K,
    ) -> List[Dict[int, float]]:
        """Batched: send all futures' prompts in one request to vLLM."""
        prompts = [
            build_translation_probe_prompt(tokenizer, fs, target_prefix)
            for fs in full_sources
        ]
        payload = {
            "model": self.model,
            "prompt": prompts,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": top_k,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        data = _http_json(
            f"{self.api_base}/completions",
            payload=payload,
            timeout=self.timeout,
        )
        choices = data.get("choices", [])
        results: List[Dict[int, float]] = []
        for i in range(len(prompts)):
            if i >= len(choices):
                results.append({})
                continue
            choice = choices[i]
            logprobs = choice.get("logprobs", {}) if isinstance(choice, dict) else {}
            dist, _ = _parse_completion_top_logprobs(
                logprobs.get("top_logprobs"), tokenizer=tokenizer,
            )
            results.append(filter_distribution(tokenizer, dist))
        return results

    def force_complete(
        self,
        tokenizer: Any,
        full_source: str,
        committed_text: str,
    ) -> str:
        """Force-finish translation by continuing from committed_text."""
        prompt = build_final_completion_prompt(tokenizer, full_source, committed_text)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.0,
            "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
        }
        data = _http_json(
            f"{self.api_base}/completions",
            payload=payload,
            timeout=self.timeout,
        )
        choices = data.get("choices", [])
        if not choices:
            return ""
        raw = str(choices[0].get("text", ""))
        return clean_model_text(raw)


# ── FutureLM ─────────────────────────────────────────────────────────────────

class FutureLM:
    """Small causal LM that samples K English continuations.

    Two modes:
      - **local** (default): loads the model via transformers on a local GPU.
      - **api**: calls an already-running vLLM server, avoiding duplicate GPU memory.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        api_base: Optional[str] = None,
        api_model_name: str = "qwen3-4b-base",
    ):
        self._api_base = api_base
        self._api_model_name = api_model_name

        if api_base:
            print(f"[FutureLM] Using vLLM API at {api_base} (model={api_model_name})")
            self.model = None
            self.tokenizer = None
            self.device = device
        else:
            print(f"[FutureLM] Loading {model_path} on {device}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
            self.model.eval()
            self.device = device
            print("[FutureLM] Loaded.")

    def sample(
        self,
        prefix_text: str,
        K: int = 10,
        future_words: int = 15,
        temperature: float = 0.9,
    ) -> List[str]:
        if self._api_base:
            return self._sample_api(prefix_text, K, future_words, temperature)
        return self._sample_local(prefix_text, K, future_words, temperature)

    def _sample_api(
        self, prefix_text: str, K: int, future_words: int, temperature: float,
    ) -> List[str]:
        import requests
        futures: List[str] = []
        seen: set = set()

        resp = requests.post(
            f"{self._api_base}/completions",
            json={
                "model": self._api_model_name,
                "prompt": prefix_text,
                "max_tokens": future_words * 3,
                "temperature": max(temperature, 1.0),
                "top_p": 0.98,
                "n": K,
                "stop": ["\n", '"', "\u201d"],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        for choice in data.get("choices", []):
            cont = choice.get("text", "").strip()
            cont = clean_model_text(cont)
            if cont and cont[-1] in ".!?":
                cont = cont[:-1]
            if not cont or re.search(r"[\u4e00-\u9fff]", cont):
                continue
            key = cont.lower()
            if key not in seen:
                seen.add(key)
                futures.append(cont)

        while len(futures) < K:
            futures.append("")
        return futures[:K]

    @torch.no_grad()
    def _sample_local(
        self, prefix_text: str, K: int, future_words: int, temperature: float,
    ) -> List[str]:
        encoded = self.tokenizer(
            prefix_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        eos_id = self.tokenizer.eos_token_id or 0
        stop_ids = [eos_id]
        for tok in ["<|im_end|>", "<|endoftext|>", "\n"]:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                stop_ids.append(tid)
        stop_ids = list(set(stop_ids))

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=future_words,
            do_sample=True,
            temperature=max(temperature, 1.0),
            top_p=0.98,
            top_k=50,
            num_return_sequences=K,
            pad_token_id=self.tokenizer.pad_token_id or eos_id,
            eos_token_id=stop_ids,
        )
        futures: List[str] = []
        seen: set = set()
        for seq in outputs:
            cont_ids = seq[prompt_len:].tolist()
            cont = self.tokenizer.decode(cont_ids, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
            cont = clean_model_text(cont)
            cont = re.split(r'[\n"\u201d]', cont)[0].strip()
            if cont and cont[-1] in ".!?":
                cont = cont[:-1]
            if not cont or re.search(r"[\u4e00-\u9fff]", cont):
                continue
            key = cont.lower()
            if key not in seen:
                seen.add(key)
                futures.append(cont)
        while len(futures) < K:
            futures.append("")
        return futures[:K]


def append_text_continuation(prefix: str, continuation: str) -> str:
    """Join observed source prefix with a future continuation."""
    if not prefix:
        return continuation
    if not continuation:
        return prefix
    if prefix[-1].isspace() or continuation[0].isspace():
        return prefix + continuation
    if continuation[0] in ",.!?;:)]}\"'":
        return prefix + continuation
    return prefix + " " + continuation


# ── Main engine ───────────────────────────────────────────────────────────────

class FutureTokenConsensusEngine:
    """Build a future-aware delta via hard next-token intersection.

    Follows the design of consensus_decoding.py:
    - Uses completion endpoint with prefix forcing
    - Filters distributions before consensus
    - Batches all futures in one API call
    - Manages a token-id pending buffer with tail trimming
    """

    def __init__(
        self,
        future_lm: FutureLM,
        vllm: VLLMCompletionClient,
        tokenizer_path: str,
        num_futures: int = 10,
        future_words: int = 15,
        future_temperature: float = 0.9,
        top_logprobs: int = TOP_K,
        max_consensus_steps: int = 6,
    ):
        self.future_lm = future_lm
        self.vllm = vllm
        self.num_futures = num_futures
        self.future_words = future_words
        self.future_temperature = future_temperature
        self.top_logprobs = top_logprobs
        self.max_consensus_steps = max_consensus_steps
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True,
        )

    def build_consensus_delta(
        self,
        src_text: str,
        committed_zh: str = "",
    ) -> Tuple[str, dict]:
        """Return a safe delta string plus a structured trace record."""
        # Step 1: Sample futures
        futures = self.future_lm.sample(
            src_text,
            K=self.num_futures,
            future_words=self.future_words,
            temperature=self.future_temperature,
        )
        # Filter to unique, non-empty futures
        unique_futures = []
        seen: set = set()
        for f in futures:
            if f and f.lower() not in seen:
                seen.add(f.lower())
                unique_futures.append(f)

        trace: dict = {
            "src_text": src_text,
            "committed_zh": committed_zh,
            "num_futures_requested": self.num_futures,
            "futures": futures,
            "unique_futures": unique_futures,
            "steps": [],
            "delta": "",
            "stop_reason": "",
        }

        if len(unique_futures) < 2:
            trace["stop_reason"] = "too_few_unique_futures"
            return "", trace

        # Step 2-5: Iterative consensus token growing
        pending_token_ids: List[int] = []

        for step_idx in range(self.max_consensus_steps):
            # Current target prefix = committed + decoded pending tokens
            target_prefix = committed_zh + self.tokenizer.decode(
                pending_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ) if pending_token_ids else committed_zh

            # Build full hypothetical sources for each future
            full_sources = [
                append_text_continuation(src_text, f)
                for f in unique_futures
            ]

            # Batched API call for all futures
            distributions = self.vllm.batch_get_next_token_distributions(
                tokenizer=self.tokenizer,
                full_sources=full_sources,
                target_prefix=target_prefix,
                top_k=self.top_logprobs,
            )

            step_rec: dict = {
                "step": step_idx + 1,
                "target_prefix": target_prefix,
                "per_future": [],
            }

            # Check for empty distributions
            has_empty = False
            for i, dist in enumerate(distributions):
                top_ids = topk_token_ids(dist, self.top_logprobs)
                step_rec["per_future"].append({
                    "future": unique_futures[i],
                    "num_candidates": len(dist),
                    "top_tokens": [
                        {"id": tid, "text": _single_token_text(self.tokenizer, tid),
                         "prob": round(dist[tid], 6)}
                        for tid in top_ids[:5]
                    ],
                })
                if not dist:
                    has_empty = True

            if has_empty:
                step_rec["decision"] = "STOP"
                step_rec["reason"] = "empty_distribution"
                trace["steps"].append(step_rec)
                trace["stop_reason"] = "empty_distribution"
                break

            # Consensus
            token_id, meta = choose_consensus_token(distributions)
            step_rec["consensus_meta"] = meta

            if token_id is None:
                step_rec["decision"] = "READ"
                step_rec["reason"] = "empty_intersection"
                trace["steps"].append(step_rec)
                trace["stop_reason"] = "empty_intersection"
                break

            token_text = _single_token_text(self.tokenizer, token_id)
            pending_token_ids.append(token_id)

            step_rec["selected_token_id"] = token_id
            step_rec["selected_text"] = token_text
            step_rec["selected_avg_prob"] = meta.get("avg_prob", 0.0)
            step_rec["decision"] = "APPEND"
            trace["steps"].append(step_rec)
        else:
            trace["stop_reason"] = "max_consensus_steps"

        # Step 6-7: Sanitize and trim pending tokens
        trimmed_ids, delta = sanitize_and_trim_pending(
            self.tokenizer, committed_zh, pending_token_ids,
        )
        trace["pending_token_ids_raw"] = pending_token_ids
        trace["pending_token_ids_trimmed"] = trimmed_ids
        trace["delta"] = delta
        if not trace["stop_reason"]:
            trace["stop_reason"] = "no_delta" if not delta else "ok"
        return delta, trace
