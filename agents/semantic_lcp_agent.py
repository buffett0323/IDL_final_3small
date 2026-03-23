"""
Semantic LCP Simultaneous Translation Agent (EN→ZH)

Pipeline per commit step:
  1. Qwen3-4B-Base (local)      → generate K English future continuations
  2. Qwen30B-Instruct (vLLM)    → for each future, translate the observed prefix
                                   using the future as context
  3. Quorum LCP (code)          → find the longest Chinese prefix that ≥60% of
                                   the K candidates agree on
  4. Emit new characters        → output consensus beyond already-committed text

Key idea: using the future as translation context gives the model enough
information to produce a consistent, accurate prefix translation.
The quorum LCP ensures we only commit what the model is confident about
regardless of how the sentence continues.

Usage:
  # 1. Start vLLM server (on GPU 0, or adjust CUDA_VISIBLE_DEVICES)
  #    (see scripts/serve_qwen30b.sh)

  # 2. Run simuleval
  simuleval \\
    --agent agents/semantic_lcp_agent.py \\
    --source data/enzh/rand100_source.txt \\
    --target data/enzh/rand100_target.txt \\
    --output outputs/semantic_lcp_k5/ \\
    --wait-k 5 \\
    --num-futures 4 \\
    --future-lm-path /data/user_data/haolingp/models/Qwen3-4B-Base \\
    --vllm-api-base http://localhost:8100/v1 \\
    --vllm-model-name qwen30b-instruct
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import contextlib
import unicodedata
from pathlib import Path
from typing import Optional

import torch

try:
    from simuleval import entrypoint
except ImportError:
    from simuleval.utils import entrypoint
from simuleval.agents.agent import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.evaluator.instance import Instance

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from model_utils import split_chinese_chars

# ── SimulEval monkey-patches (same as other agents) ──────────────────────────
_original_summarize = Instance.summarize


def _patched_summarize(self):
    result = _original_summarize(self)
    result["metric"] = self.metrics
    return result


Instance.summarize = _patched_summarize

from simuleval.evaluator.scorers.latency_scorer import LatencyScorer
from statistics import mean


def _patched_scorer_call(self, instances):
    scores = []
    for index, ins in instances.items():
        delays = getattr(ins, self.timestamp_type)
        if not delays or ins.prediction_length == 0:
            continue
        if ins.source_length == 0:
            continue
        try:
            score = self.compute(ins)
        except ZeroDivisionError:
            continue
        ins.metrics[self.metric_name] = score
        scores.append(score)
    return mean(scores) if scores else 0.0


LatencyScorer.__call__ = _patched_scorer_call

from simuleval.evaluator.evaluator import SentenceLevelEvaluator
from simuleval.data.dataloader.dataloader import IterableDataloader


def _patched_eval_call(self, system):
    iterator = getattr(self, "iterator", None)
    if iterator is None:
        iterator = self.maybe_tqdm(self.instances.values())
    with open(
        self.output / "instances.log", "a"
    ) if self.output else contextlib.nullcontext() as file:
        system.reset()
        for sample in iterator:
            instance = (
                self.instance_class(
                    self.dataloader.cur_index, self.dataloader, self.args
                )
                if isinstance(self.dataloader, IterableDataloader)
                else sample
            )
            while not self.is_finished(instance):
                input_segment = instance.send_source(self.source_segment_size)
                output_segment = system.pushpop(input_segment)
                instance.receive_prediction(output_segment)
                if instance.finish_prediction:
                    system.reset()
            if not self.score_only and self.output:
                file.write(json.dumps(instance.summarize()) + "\n")

    if self.output:
        self.build_instances_from_log()

    results = self.results
    if self.output:
        with open(self.output / "scores", "w") as f:
            f.write(results.to_string())

    print(results.to_string(index=False))


SentenceLevelEvaluator.__call__ = _patched_eval_call


# ── Chinese text utilities ────────────────────────────────────────────────────

def normalize_zh(text: str) -> str:
    """Normalize Chinese text: strip spaces, normalize unicode."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", "", text).strip()


# Characters that are safe word/phrase boundaries in Chinese
_ZH_BOUNDARIES = frozenset("，。！？；：、…—""''「」【】\n")


def longest_prefix_with_quorum(candidates: list[str], K: int) -> str:
    """Return the longest char-level prefix shared by at least K candidates,
    truncated at the last safe word boundary to avoid mid-word commits.

    A 'safe boundary' is any punctuation character or the end of a common
    multi-character Chinese morpheme pattern (2-4 chars).  If the raw quorum
    prefix ends mid-word, we walk back to the nearest boundary character.
    """
    if not candidates or K <= 0:
        return ""
    prefix_count: dict[str, int] = {}
    for c in candidates:
        p = ""
        for ch in c:
            p += ch
            prefix_count[p] = prefix_count.get(p, 0) + 1
    best = ""
    for p, cnt in prefix_count.items():
        if cnt >= K and len(p) > len(best):
            best = p
    if not best:
        return ""

    # Truncate at last safe boundary so we never commit mid-word.
    # Walk backwards from end of best to find a boundary or complete 2-char unit.
    # Rule: if the last char is a punctuation boundary → safe as-is.
    #       Otherwise, trim back to the last punctuation boundary.
    #       If no punctuation boundary exists, emit nothing (wait for more context).
    if best[-1] in _ZH_BOUNDARIES:
        return best  # ends at punctuation — always safe

    # Find last boundary index
    last_boundary = -1
    for i, ch in enumerate(best):
        if ch in _ZH_BOUNDARIES:
            last_boundary = i

    if last_boundary >= 0:
        # Truncate to include the boundary character
        return best[: last_boundary + 1]

    # No punctuation boundary: only commit if we have ≥2 chars (likely a
    # complete morpheme) AND the prefix length is a multiple of 2 (common
    # Chinese word length).  Otherwise be conservative and wait.
    if len(best) >= 2:
        # Commit up to the nearest even length (2-char word boundary heuristic)
        safe_len = (len(best) // 2) * 2
        return best[:safe_len] if safe_len > 0 else ""

    return ""  # single ambiguous char — don't commit yet


def get_quorum_lcp(
    committed: str,
    candidates: list[str],
    consensus_ratio: float = 0.6,
) -> str:
    """Return the quorum LCP delta beyond the already-committed text.

    committed      : Chinese text already emitted.
    candidates     : K Chinese prefix translations (full, not delta).
    consensus_ratio: fraction that must agree (default 0.6 = 60%).

    Returns the new characters to commit (empty string = READ).
    """
    committed_norm = normalize_zh(committed)
    deltas: list[str] = []
    for c in candidates:
        c_norm = normalize_zh(c)
        if committed_norm and c_norm.startswith(committed_norm):
            deltas.append(c_norm[len(committed_norm):])
        elif not committed_norm:
            deltas.append(c_norm)
        # If candidate doesn't start with committed, skip (inconsistent)
    if not deltas:
        return ""
    M = len(deltas)
    K = max(1, math.ceil(consensus_ratio * M))
    return longest_prefix_with_quorum([d for d in deltas if d], K)


# ── vLLM API client ───────────────────────────────────────────────────────────

class VLLMClient:
    """Thin wrapper around OpenAI-compatible vLLM API."""

    def __init__(self, api_base: str, model_name: str, timeout: float = 60.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key="EMPTY", timeout=timeout)
        self.model = model_name

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def translate_prefix_with_context(
        self,
        observed_prefix: str,
        future_continuation: str,
        committed_zh: str,
    ) -> str:
        """Ask Qwen30B to translate the observed prefix given future context.

        The future is a hint to improve translation quality — especially for
        ambiguous words (e.g. names, roles) whose correct translation depends
        on what comes next.  We ask the model to translate ONLY the confirmed
        observed part, not the future.
        """
        already = f'\n(Already committed: "{committed_zh}")' if committed_zh else ""
        system = (
            "You are a professional simultaneous English-to-Chinese translator. "
            "Translate ONLY the [observed] part to Simplified Chinese. "
            "Use the [future context] to improve your translation quality. "
            "Output ONLY the Chinese translation of the observed part, nothing else."
        )
        user = (
            f"[observed]: {observed_prefix}\n"
            f"[future context (do not translate)]: {future_continuation}\n"
            f"{already}\n"
            "Chinese translation of [observed] only:"
        )
        result = self.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=128,
            temperature=0.0,
        )
        return normalize_zh(result)


# ── Future LM (Qwen3-4B-Base, local) ─────────────────────────────────────────

class FutureLM:
    """Small local causal LM that samples K English continuations."""

    def __init__(self, model_path: str, device: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[FutureLM] Loading {model_path} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
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

    @torch.no_grad()
    def sample(
        self,
        prefix_text: str,
        K: int = 4,
        future_words: int = 15,
        temperature: float = 0.9,
    ) -> list[str]:
        """Return K plausible English continuations of prefix_text."""
        prompt = (
            f"Continue this English news sentence naturally with about "
            f"{future_words} more words:\n\n\"{prefix_text}"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        eos_id = self.tokenizer.eos_token_id or 0
        stop_ids = [eos_id]
        for tok in ["<|im_end|>", "<|endoftext|>", "\n"]:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                stop_ids.append(tid)
        stop_ids = list(set(stop_ids))

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=future_words,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=K,
            pad_token_id=self.tokenizer.pad_token_id or eos_id,
            eos_token_id=stop_ids,
            repetition_penalty=1.1,
        )
        futures: list[str] = []
        seen: set[str] = set()
        for seq in outputs:
            cont = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            cont = re.split(r'[\n"\u201d]', cont)[0].strip()
            if cont and cont[-1] in ".!?":
                cont = cont[:-1]
            full = (prefix_text + " " + cont).strip()
            if full not in seen:
                seen.add(full)
                futures.append(full)
        while len(futures) < K:
            futures.append(prefix_text)
        return futures[:K]


# ── Agent ─────────────────────────────────────────────────────────────────────

@entrypoint
class SemanticLCPAgent(TextToTextAgent):
    """Simultaneous EN→ZH agent using LM-sampled futures + semantic LCP."""

    def __init__(self, args):
        super().__init__(args)

        self.wait_k = args.wait_k
        self.num_futures = args.num_futures
        self.future_words = args.future_words
        self.future_temperature = args.future_temperature
        self.consensus_ratio = args.consensus_ratio
        self.device = f"cuda:{args.base_gpu}" if torch.cuda.is_available() else "cpu"

        # Future LM: Qwen3-4B-Base (local)
        # When num_futures == 0 (direct-translate mode), future LM is not used.
        if self.num_futures > 0:
            self._future_lm = FutureLM(args.future_lm_path, self.device)
        else:
            self._future_lm = None
            print("[SemanticLCP] num_futures=0: direct-translate mode (no future LM loaded)")

        # vLLM client: Qwen30B-Instruct (served separately)
        print(f"[SemanticLCP] Connecting to vLLM at {args.vllm_api_base}")
        self._vllm = VLLMClient(
            api_base=args.vllm_api_base,
            model_name=args.vllm_model_name,
        )

        # Per-sentence state (reset between sentences)
        self._committed: str = ""       # Chinese text already written
        self._pending: list[str] = []   # buffered chars waiting to be emitted
        self._consensus_cache: dict[str, str] = {}  # src_text → consensus
        self._sentence_id: int = -1

        # Trace log
        self._trace_path: Optional[Path] = None
        if getattr(args, "output", None):
            self._trace_path = Path(args.output) / "lcp_trace.jsonl"
            if self._trace_path.exists():
                self._trace_path.unlink()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-k", type=int, default=5)
        parser.add_argument(
            "--future-lm-path", type=str,
            default="/data/user_data/haolingp/models/Qwen3-4B-Base",
            help="Local path to Qwen3-4B-Base for English future generation.",
        )
        parser.add_argument(
            "--vllm-api-base", type=str, default="http://localhost:8100/v1",
            help="Base URL of the running vLLM server (Qwen30B-Instruct).",
        )
        parser.add_argument(
            "--vllm-model-name", type=str, default="qwen30b-instruct",
            help="Model name registered in the vLLM server.",
        )
        parser.add_argument("--num-futures", type=int, default=4,
                            help="K: number of English futures to sample.")
        parser.add_argument("--future-words", type=int, default=15,
                            help="Max new tokens per future continuation.")
        parser.add_argument("--future-temperature", type=float, default=0.9,
                            help="Sampling temperature for future LM.")
        parser.add_argument("--consensus-ratio", type=float, default=0.6,
                            help="Fraction of K candidates that must agree for LCP commit.")
        parser.add_argument("--base-gpu", type=int, default=0,
                            help="GPU index for Qwen3-4B future LM.")

    def reset(self):
        super().reset()
        self._committed = ""
        self._pending = []
        self._consensus_cache = {}
        self._sentence_id = getattr(self, "_sentence_id", -1) + 1

    # ── Core policy ──────────────────────────────────────────────────────────

    def policy(self):
        src_len = len(self.states.source)

        # Emit buffered chars first (from previous consensus)
        if self._pending:
            ch = self._pending.pop(0)
            finished = self.states.source_finished and not self._pending
            return WriteAction(ch, finished=finished)

        # Wait-k: read until we have enough source context
        if not self.states.source_finished and src_len < self.wait_k:
            return ReadAction()

        # At end of source: force-translate whatever is left
        if self.states.source_finished:
            return self._force_finish()

        # Compute semantic consensus for current source prefix
        src_text = " ".join(self.states.source)
        new_chars = self._get_consensus_delta(src_text)

        if new_chars:
            # Queue all new chars; emit the first one now
            units = split_chinese_chars(new_chars)
            if units:
                self._committed += new_chars
                self._pending = list("".join(units[1:]))
                return WriteAction(units[0], finished=False)

        # No consensus yet → read more
        return ReadAction()

    # ── Semantic consensus ────────────────────────────────────────────────────

    def _get_consensus_delta(self, src_text: str) -> str:
        """Return new Chinese chars to commit for current source prefix.

        Two modes:
          num_futures == 0  (direct-translate):
            Ask Qwen30B to translate the observed prefix directly.
            Commit ALL new characters beyond _committed.
            This is the fair Qwen30B baseline — no future sampling, no consensus.

          num_futures > 0  (semantic LCP):
            Generate K futures, translate each with Qwen30B, run quorum LCP.
            Only commit what ≥consensus_ratio of candidates agree on.
        """
        if src_text in self._consensus_cache:
            cached = self._consensus_cache[src_text]
            committed_norm = normalize_zh(self._committed)
            if cached.startswith(committed_norm) and len(cached) > len(committed_norm):
                return cached[len(committed_norm):]
            return ""

        # ── Direct-translate mode (num_futures == 0) ─────────────────────────
        if self.num_futures == 0:
            try:
                zh = self._vllm.translate_prefix_with_context(
                    observed_prefix=src_text,
                    future_continuation="",
                    committed_zh=self._committed,
                )
            except Exception as e:
                print(f"[WARN] vLLM call failed: {e}")
                return ""

            committed_norm = normalize_zh(self._committed)
            zh_norm = normalize_zh(zh)
            if zh_norm.startswith(committed_norm):
                new_delta = zh_norm[len(committed_norm):]
            else:
                new_delta = zh_norm  # inconsistent — take full translation

            self._consensus_cache[src_text] = committed_norm + new_delta
            self._trace(src_text, [], [zh_norm], new_delta, 1, 1)
            return new_delta

        # ── Semantic LCP mode (num_futures > 0) ──────────────────────────────
        # 1. Generate K English futures
        futures = self._future_lm.sample(
            src_text,
            K=self.num_futures,
            future_words=self.future_words,
            temperature=self.future_temperature,
        )

        # 2. Translate prefix once per future via Qwen30B
        candidates: list[str] = []
        for future in futures:
            continuation = future[len(src_text):].strip() if future.startswith(src_text) else future
            try:
                zh = self._vllm.translate_prefix_with_context(
                    observed_prefix=src_text,
                    future_continuation=continuation,
                    committed_zh=self._committed,
                )
                if zh:
                    candidates.append(zh)
            except Exception as e:
                print(f"[WARN] vLLM call failed: {e}")

        if not candidates:
            return ""

        # 3. Quorum LCP: find what ≥consensus_ratio agree on beyond committed
        M = len(candidates)
        K = max(1, math.ceil(self.consensus_ratio * M))
        new_delta = get_quorum_lcp(self._committed, candidates, self.consensus_ratio)

        consensus_full = normalize_zh(self._committed) + new_delta
        self._consensus_cache[src_text] = consensus_full
        self._trace(src_text, futures, candidates, new_delta, M, K)
        return new_delta

    def _force_finish(self) -> WriteAction | ReadAction:
        """At source EOS, translate the full observed source and emit remainder."""
        src_text = " ".join(self.states.source)
        if src_text in self._consensus_cache:
            full = self._consensus_cache[src_text]
        else:
            # Single translation of full source (no futures needed at EOS)
            try:
                full = self._vllm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional English-to-Chinese translator. "
                                "Translate the following English sentence to Simplified Chinese. "
                                "Output ONLY the Chinese translation."
                            ),
                        },
                        {"role": "user", "content": src_text},
                    ],
                    max_tokens=256,
                    temperature=0.0,
                )
                full = normalize_zh(full)
            except Exception as e:
                print(f"[WARN] Force-finish vLLM call failed: {e}")
                full = self._committed
            self._consensus_cache[src_text] = full

        committed_norm = normalize_zh(self._committed)
        if full.startswith(committed_norm) and len(full) > len(committed_norm):
            remainder = full[len(committed_norm):]
        else:
            remainder = full

        if remainder:
            units = split_chinese_chars(remainder)
            if units:
                self._committed += remainder
                self._pending = list("".join(units[1:]))
                return WriteAction(units[0], finished=not self._pending)

        return WriteAction("", finished=True)

    def _trace(
        self,
        src_text: str,
        futures: list[str],
        candidates: list[str],
        delta: str,
        M: int,
        K: int,
    ):
        if self._trace_path is None:
            return
        record = {
            "sentence_id": self._sentence_id,
            "src_text": src_text,
            "committed": self._committed,
            "futures": futures,
            "candidates": candidates,
            "quorum_K": K,
            "quorum_M": M,
            "delta": delta,
        }
        with self._trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
