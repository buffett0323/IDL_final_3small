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

  # 2. Run simuleval (add --verbose for step-by-step stdout; JSONL at OUTPUT/lcp_trace.jsonl)
  simuleval \\
    --agent agents/semantic_lcp_agent.py \\
    --source data/enzh/rand100_source.txt \\
    --target data/enzh/rand100_target.txt \\
    --output outputs/semantic_lcp_k5/ \\
    --wait-k 5 \\
    --num-futures 4 \\
    --verbose \\
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
import simuleval.evaluator.scorers.latency_scorer as _latency_scorer_module
import statistics as _statistics
from statistics import mean


def _safe_latency_mean(seq):
    """SimulEval ATDScorer (and ATD compute) call mean() on possibly empty lists."""
    return _statistics.mean(seq) if seq else 0.0


_latency_scorer_module.mean = _safe_latency_mean


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


def get_delta_beyond_committed(committed: str, candidate: str) -> str:
    """Return the candidate suffix beyond the already committed prefix."""
    committed_norm = normalize_zh(committed)
    cand_norm = normalize_zh(candidate)
    if committed_norm and cand_norm.startswith(committed_norm):
        return cand_norm[len(committed_norm):]
    if not committed_norm:
        return cand_norm
    return cand_norm


def _js_divergence(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence for two categorical distributions."""
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def _kl(a: list[float], b: list[float]) -> float:
        out = 0.0
        for ai, bi in zip(a, b):
            if ai > 0.0 and bi > 0.0:
                out += ai * math.log(ai / bi)
        return out

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def avg_pairwise_js_over_candidate_deltas(
    committed: str,
    candidates: list[str],
    steps: int = 3,
) -> float:
    """Approximate DD on Qwen candidates via next-char empirical disagreement.

    Each future-conditioned Chinese candidate induces a one-hot distribution over
    the next character beyond the already-committed prefix. We compute average
    pairwise JS across the first `steps` positions. This is a string-level analog
    of the NLLB DD gate: higher JS means plausible futures disagree earlier.
    """
    if len(candidates) < 2 or steps <= 0:
        return 0.0

    deltas = [get_delta_beyond_committed(committed, cand) for cand in candidates]
    scores: list[float] = []
    eos = "<eos>"

    for pos in range(steps):
        symbols = []
        for delta in deltas:
            symbols.append(delta[pos] if pos < len(delta) else eos)
        vocab = sorted(set(symbols))
        if len(vocab) <= 1:
            scores.append(0.0)
            continue

        dists: list[list[float]] = []
        for sym in symbols:
            dist = [0.0] * len(vocab)
            dist[vocab.index(sym)] = 1.0
            dists.append(dist)

        pair_scores: list[float] = []
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                pair_scores.append(_js_divergence(dists[i], dists[j]))
        if pair_scores:
            scores.append(sum(pair_scores) / len(pair_scores))

    return sum(scores) / len(scores) if scores else 0.0


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

    def force_complete(
        self,
        tokenizer,
        full_source: str,
        committed_text: str,
    ) -> str:
        """Force-finish by continuing from committed_text via completion endpoint.

        Uses apply_chat_template + assistant prefix forcing so the model MUST
        continue from the committed text (no duplication possible).
        """
        import json as _json
        import urllib.request
        import urllib.error

        if not str(committed_text or "").strip():
            messages = [{
                "role": "user",
                "content": (
                    "[TASK]\nTranslate the [INPUT] text into Chinese.\n\n"
                    f"[INPUT]\n{full_source}\n\n"
                    "[IMPORTANT]\nOutput the complete Chinese translation only."
                ),
            }]
        else:
            messages = [{
                "role": "user",
                "content": (
                    "[TASK]\nTranslate the [INPUT] text into Chinese.\n\n"
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
        if committed_text:
            prompt += committed_text

        # Use completion endpoint (not chat) for prefix forcing
        base = str(self.client.base_url).rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.0,
            "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
        }
        req = urllib.request.Request(
            f"{base}/completions",
            data=_json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            body = resp.read().decode("utf-8")
            data = _json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            return ""
        raw = str(choices[0].get("text", ""))
        # Clean special tokens from output
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        raw = re.sub(r"<think>.*$", "", raw, flags=re.DOTALL | re.IGNORECASE)
        raw = raw.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
        return raw


# ── Future LM (Qwen3-4B-Base, local or vLLM API) ─────────────────────────────

class FutureLM:
    """Small causal LM that samples K English continuations.

    Two modes:
      - **local** (default): loads the model via transformers on a local GPU.
      - **api**: calls an already-running vLLM server, avoiding duplicate GPU memory.

    The api_base / api_model_name constructor args select the mode:
    if api_base is given, the local model is never loaded.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        api_base: str | None = None,
        api_model_name: str | None = None,
    ):
        self._api_base = api_base
        self._api_model_name = api_model_name or "qwen3-4b-base"

        if api_base:
            # ── API mode: no local model needed ──
            print(f"[FutureLM] Using vLLM API at {api_base} (model={self._api_model_name})")
            self.model = None
            self.tokenizer = None
            self.device = device
        else:
            # ── Local mode: load onto GPU ──
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

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample(
        self,
        prefix_text: str,
        K: int = 4,
        future_words: int = 15,
        temperature: float = 0.9,
    ) -> list[str]:
        """Return K plausible English continuations of prefix_text."""
        if self._api_base:
            return self._sample_api(prefix_text, K, future_words, temperature)
        return self._sample_local(prefix_text, K, future_words, temperature)

    # ── API mode ─────────────────────────────────────────────────────────────

    def _sample_api(
        self, prefix_text: str, K: int, future_words: int, temperature: float,
    ) -> list[str]:
        import requests

        futures: list[str] = []
        seen: set[str] = set()

        # Match the local-mode instruction prompt so the base model
        # generates plausible English continuations, not random text.
        instruction_prompt = (
            f"Continue this English news sentence naturally with about "
            f"{future_words} more words:\n\n\"{prefix_text}"
        )

        # Sample K completions in one batched request via vLLM /completions
        resp = requests.post(
            f"{self._api_base}/completions",
            json={
                "model": self._api_model_name,
                "prompt": instruction_prompt,
                "max_tokens": future_words * 3,  # token budget (subword → ~3 tokens/word)
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
            if cont and cont[-1] in ".!?":
                cont = cont[:-1]
            full = (prefix_text + " " + cont).strip() if cont else prefix_text
            if full not in seen:
                seen.add(full)
                futures.append(full)

        while len(futures) < K:
            futures.append(prefix_text)
        return futures[:K]

    # ── Local mode ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _sample_local(
        self, prefix_text: str, K: int, future_words: int, temperature: float,
    ) -> list[str]:
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
            temperature=max(temperature, 1.0),
            top_p=0.98,
            num_return_sequences=K,
            pad_token_id=self.tokenizer.pad_token_id or eos_id,
            eos_token_id=stop_ids,
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
        self.gate_js = args.gate_js
        self.gate_tau = args.gate_tau
        self.gate_steps = args.gate_steps
        self.device = f"cuda:{args.base_gpu}" if torch.cuda.is_available() else "cpu"

        if self.gate_js and self.num_futures <= 0:
            raise ValueError("--gate-js requires --num-futures > 0.")

        # Future LM: Qwen3-4B-Base (local or vLLM API)
        # When num_futures == 0 (direct-translate mode), future LM is not used.
        future_api = getattr(args, "future_lm_api", None)
        future_api_name = getattr(args, "future_lm_api_model", None)
        if self.num_futures > 0:
            self._future_lm = FutureLM(
                args.future_lm_path, self.device,
                api_base=future_api, api_model_name=future_api_name,
            )
        else:
            self._future_lm = None
            print("[SemanticLCP] num_futures=0: direct-translate mode (no future LM loaded)")

        # vLLM client: Qwen30B-Instruct (served separately)
        print(f"[SemanticLCP] Connecting to vLLM at {args.vllm_api_base}")
        self._vllm = VLLMClient(
            api_base=args.vllm_api_base,
            model_name=args.vllm_model_name,
        )

        # Tokenizer for force_complete (prefix-forcing via completion endpoint)
        from transformers import AutoTokenizer
        vllm_tokenizer_path = getattr(args, "vllm_tokenizer_path", None)
        if not vllm_tokenizer_path:
            vllm_tokenizer_path = "/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8"
        self._instruct_tokenizer = AutoTokenizer.from_pretrained(
            vllm_tokenizer_path, trust_remote_code=True,
        )

        # Per-sentence state (reset between sentences)
        self._committed: str = ""       # Chinese text already written
        self._pending: list[str] = []   # buffered chars waiting to be emitted
        self._consensus_cache: dict[str, str] = {}  # src_text → consensus
        self._sentence_id: int = -1

        # Trace log (JSONL under --output) + optional --verbose stdout
        self._verbose: bool = getattr(args, "verbose", False)
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
            "--future-lm-api", type=str, default=None,
            help="vLLM API base for future LM (e.g. http://localhost:8102/v1). "
                 "If set, uses the API instead of loading the model locally.",
        )
        parser.add_argument(
            "--future-lm-api-model", type=str, default="qwen3-4b-base",
            help="Model name on the future-LM vLLM server.",
        )
        parser.add_argument(
            "--vllm-api-base", type=str, default="http://localhost:8100/v1",
            help="Base URL of the running vLLM server (Qwen30B-Instruct).",
        )
        parser.add_argument(
            "--vllm-model-name", type=str, default="qwen30b-instruct",
            help="Model name registered in the vLLM server.",
        )
        parser.add_argument(
            "--vllm-tokenizer-path", type=str,
            default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8",
            help="Local tokenizer path for the vLLM-served model (for prefix-forcing).",
        )
        parser.add_argument("--num-futures", type=int, default=4,
                            help="K: number of English futures to sample.")
        parser.add_argument("--future-words", type=int, default=15,
                            help="Max new tokens per future continuation.")
        parser.add_argument("--future-temperature", type=float, default=0.9,
                            help="Sampling temperature for future LM.")
        parser.add_argument("--consensus-ratio", type=float, default=0.6,
                            help="Fraction of K candidates that must agree for LCP commit.")
        parser.add_argument(
            "--gate-js",
            action="store_true",
            help=(
                "Use future-conditioned disagreement as a READ/WRITE gate on top of "
                "Qwen direct translation. This is a Qwen-side DD-style baseline: "
                "if avg pairwise JS over candidate next-char deltas > gate_tau, READ more."
            ),
        )
        parser.add_argument(
            "--gate-tau", type=float, default=0.15,
            help="Threshold for --gate-js. READ if avg JS > gate_tau.",
        )
        parser.add_argument(
            "--gate-steps", type=int, default=3,
            help="How many next-char positions beyond committed to include in Qwen-side JS gating.",
        )
        parser.add_argument("--base-gpu", type=int, default=0,
                            help="GPU index for Qwen3-4B future LM.")
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print each consensus step (English futures, ZH candidates, LCP delta) to stdout.",
        )

    def reset(self):
        super().reset()
        self._committed = ""
        self._pending = []
        self._consensus_cache = {}
        self._sentence_id = getattr(self, "_sentence_id", -1) + 1

    # ── Core policy ──────────────────────────────────────────────────────────

    def policy(self):
        src_len = len(self.states.source)

        # Emit buffered chars first — never set finished=True here;
        # let _force_finish handle the final completion after source ends.
        if self._pending:
            ch = self._pending.pop(0)
            return WriteAction(ch, finished=False)

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

          gate_js == True   (Qwen DD-style gate):
            Generate K futures, translate each with Qwen30B, compute an empirical
            next-char JS disagreement score across future-conditioned candidates.
            If disagreement is high, READ. Otherwise commit the direct Qwen delta.

          num_futures > 0 and gate_js == False  (semantic LCP):
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

        # ── Future-aware modes (num_futures > 0) ─────────────────────────────
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

        M = len(candidates)

        if self.gate_js:
            try:
                direct_zh = self._vllm.translate_prefix_with_context(
                    observed_prefix=src_text,
                    future_continuation="",
                    committed_zh=self._committed,
                )
            except Exception as e:
                print(f"[WARN] vLLM direct baseline call failed: {e}")
                return ""

            direct_delta = get_delta_beyond_committed(self._committed, direct_zh)
            js_score = avg_pairwise_js_over_candidate_deltas(
                committed=self._committed,
                candidates=candidates,
                steps=self.gate_steps,
            )
            gate_decision = "READ" if js_score > self.gate_tau else "COMMIT"
            new_delta = "" if gate_decision == "READ" else direct_delta

            self._consensus_cache[src_text] = normalize_zh(self._committed) + new_delta
            self._trace(
                src_text,
                futures,
                candidates,
                new_delta,
                M,
                0,
                mode="qwen_dd_gate",
                direct_candidate=normalize_zh(direct_zh),
                gate_js_score=js_score,
                gate_tau=self.gate_tau,
                gate_steps=self.gate_steps,
                gate_decision=gate_decision,
            )
            return new_delta

        # 3. Quorum LCP: find what ≥consensus_ratio agree on beyond committed
        K = max(1, math.ceil(self.consensus_ratio * M))
        new_delta = get_quorum_lcp(self._committed, candidates, self.consensus_ratio)

        consensus_full = normalize_zh(self._committed) + new_delta
        self._consensus_cache[src_text] = consensus_full
        self._trace(
            src_text,
            futures,
            candidates,
            new_delta,
            M,
            K,
            mode="semantic_lcp",
        )
        return new_delta

    def _force_finish(self) -> WriteAction:
        """At source EOS, continue from committed prefix (no duplication)."""
        src_text = " ".join(self.states.source)

        try:
            continuation = self._vllm.force_complete(
                tokenizer=self._instruct_tokenizer,
                full_source=src_text,
                committed_text=self._committed,
            )
        except Exception as e:
            print(f"[WARN] Force-finish vLLM call failed: {e}")
            continuation = ""

        if continuation:
            self._committed += continuation
            units = split_chinese_chars(continuation)
            if units:
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
        **extra,
    ):
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
        record.update(extra)
        if self._verbose:
            print(self._format_trace(record), flush=True)
        if self._trace_path is not None:
            with self._trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _format_trace(record: dict) -> str:
        lines = [
            "── SemanticLCP trace ──",
            f"sentence_id={record['sentence_id']} quorum={record['quorum_K']}/{record['quorum_M']}",
            f"src: {record['src_text']}",
            f"committed_zh (before delta): {record['committed']!r}",
        ]
        futs = record["futures"]
        if futs:
            lines.append("English futures:")
            for i, fut in enumerate(futs, 1):
                lines.append(f"  [{i}] {fut}")
        cands = record["candidates"]
        if cands:
            lines.append("ZH candidates:")
            for i, zh in enumerate(cands, 1):
                lines.append(f"  [{i}] {zh}")
        if "direct_candidate" in record:
            lines.append(f"Direct Qwen candidate: {record['direct_candidate']}")
        if "gate_js_score" in record:
            lines.append(
                "Gate JS: "
                f"{record['gate_js_score']:.4f} "
                f"(tau={record.get('gate_tau')}, steps={record.get('gate_steps')}) "
                f"→ {record.get('gate_decision')}"
            )
        label = "Delta (new chars)" if record.get("mode") == "qwen_dd_gate" else "LCP delta (new chars)"
        lines.append(f"{label}: {record['delta']!r}")
        lines.append("──")
        return "\n".join(lines)
