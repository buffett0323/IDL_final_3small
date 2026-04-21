"""Future-sampled hard token-intersection consensus agent for EN->ZH SiMT.

Method 3 (rewritten to match consensus_decoding.py design):
  - Uses vLLM /completions endpoint with prefix forcing (not chat)
  - Filters disallowed tokens BEFORE consensus
  - Batches all futures in one API call
  - Force-finish continues from committed prefix (no duplication)
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import torch

try:
    from simuleval import entrypoint
except ImportError:
    from simuleval.utils import entrypoint
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.agents.agent import TextToTextAgent
from simuleval.evaluator.instance import Instance

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from model_utils import split_chinese_chars
from token_consensus_core import (
    FutureLM,
    FutureTokenConsensusEngine,
    VLLMCompletionClient,
    clean_model_text,
    normalize_zh,
)

# ── SimulEval monkey-patches ────────────────────────────────────────────────
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
    return _statistics.mean(seq) if seq else 0.0


_latency_scorer_module.mean = _safe_latency_mean


def _patched_scorer_call(self, instances):
    scores = []
    for _, ins in instances.items():
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

from simuleval.data.dataloader.dataloader import IterableDataloader
from simuleval.evaluator.evaluator import SentenceLevelEvaluator


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


@entrypoint
class FutureTokenConsensusAgent(TextToTextAgent):
    """Streaming EN->ZH agent using future-sampled hard token consensus."""

    def __init__(self, args):
        super().__init__(args)

        self.wait_k = args.wait_k
        self.device = f"cuda:{args.base_gpu}" if torch.cuda.is_available() else "cpu"
        self._verbose = getattr(args, "verbose", False)

        future_api = getattr(args, "future_lm_api", None)
        future_api_name = getattr(args, "future_lm_api_model", "qwen3-4b-base")
        self._future_lm = FutureLM(
            args.future_lm_path, self.device,
            api_base=future_api, api_model_name=future_api_name,
        )
        self._vllm = VLLMCompletionClient(
            api_base=args.vllm_api_base,
            model_name=args.vllm_model_name,
            timeout=120.0,
        )
        self._engine = FutureTokenConsensusEngine(
            future_lm=self._future_lm,
            vllm=self._vllm,
            tokenizer_path=args.vllm_tokenizer_path,
            num_futures=args.num_futures,
            future_words=args.future_words,
            future_temperature=args.future_temperature,
            top_logprobs=args.top_logprobs,
            max_consensus_steps=args.max_consensus_steps,
            pool_mode=getattr(args, "pool_mode", "topk"),
            pool_p=float(getattr(args, "pool_p", 0.9)),
        )

        self._committed: str = ""
        self._pending: list[str] = []
        self._consensus_cache: dict[str, str] = {}
        self._trace_cache: dict[str, dict] = {}
        self._sentence_id: int = -1
        self._trace_path: Path | None = None
        if getattr(args, "output", None):
            self._trace_path = Path(args.output) / "token_consensus_trace.jsonl"
            if self._trace_path.exists():
                self._trace_path.unlink()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-k", type=int, default=5)
        parser.add_argument(
            "--future-lm-path",
            type=str,
            default="/data/user_data/haolingp/models/Qwen3-4B-Base",
            help="Local path to the future-sampling base LM.",
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
            "--vllm-api-base",
            type=str,
            default="http://localhost:8100/v1",
            help="Base URL of the running vLLM server.",
        )
        parser.add_argument(
            "--vllm-model-name",
            type=str,
            default="qwen30b-instruct",
            help="Model name registered in the vLLM server.",
        )
        parser.add_argument(
            "--vllm-tokenizer-path",
            type=str,
            default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8",
            help="Local tokenizer path for the vLLM-served translator.",
        )
        parser.add_argument(
            "--num-futures",
            type=int,
            default=10,
            help="Number of future samples to draw from the base LM.",
        )
        parser.add_argument("--future-words", type=int, default=15)
        parser.add_argument("--future-temperature", type=float, default=0.9)
        parser.add_argument(
            "--top-logprobs",
            type=int,
            default=10,
            help="Per-future top-k next-token candidates used for hard intersection.",
        )
        parser.add_argument(
            "--max-consensus-steps",
            type=int,
            default=6,
            help="Maximum number of consensus tokens to append at one source prefix.",
        )
        parser.add_argument("--base-gpu", type=int, default=0)
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print each consensus step to stdout.",
        )
        parser.add_argument(
            "--pool-mode", choices=["topk", "topp", "minp"], default="topk",
            help="Candidate-pool selector applied to each future's next-token "
                 "distribution BEFORE intersection. 'topk' (default) uses the "
                 "fixed top-K set; 'topp' uses nucleus sampling with --pool-p "
                 "as p; 'minp' keeps tokens with prob >= pool_p * max_prob.",
        )
        parser.add_argument(
            "--pool-p", type=float, default=0.9,
            help="p parameter for --pool-mode=topp (cumulative prob) or minp "
                 "(fraction of max-prob). Ignored for topk.",
        )

    def reset(self):
        super().reset()
        self._committed = ""
        self._pending = []
        self._consensus_cache = {}
        self._trace_cache = {}
        self._sentence_id = getattr(self, "_sentence_id", -1) + 1

    def policy(self):
        src_len = len(self.states.source)

        # Drain pending characters first — never set finished=True here;
        # let _force_finish handle the final completion after source ends.
        if self._pending:
            ch = self._pending.pop(0)
            return WriteAction(ch, finished=False)

        # Wait for k source words
        if not self.states.source_finished and src_len < self.wait_k:
            return ReadAction()

        # Source finished: force complete from committed prefix
        if self.states.source_finished:
            return self._force_finish()

        # Normal: try consensus decoding
        src_text = " ".join(self.states.source)
        new_chars = self._get_consensus_delta(src_text)
        if new_chars:
            units = split_chinese_chars(new_chars)
            if units:
                self._committed += new_chars
                self._pending = list("".join(units[1:]))
                return WriteAction(units[0], finished=False)

        return ReadAction()

    def _get_consensus_delta(self, src_text: str) -> str:
        if src_text in self._consensus_cache:
            cached = self._consensus_cache[src_text]
            committed_norm = normalize_zh(self._committed)
            if cached.startswith(committed_norm) and len(cached) > len(committed_norm):
                return cached[len(committed_norm):]
            return ""

        new_delta, trace = self._engine.build_consensus_delta(
            src_text=src_text,
            committed_zh=self._committed,
        )
        self._consensus_cache[src_text] = normalize_zh(self._committed) + new_delta
        self._trace_cache[src_text] = trace
        self._trace(src_text, trace)
        return new_delta

    def _force_finish(self) -> WriteAction:
        """Force-finish by continuing from committed prefix (no duplication)."""
        src_text = " ".join(self.states.source)

        # Try to get continuation from vLLM, continuing from committed prefix
        try:
            continuation = self._vllm.force_complete(
                tokenizer=self._engine.tokenizer,
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

    def _trace(self, src_text: str, trace: dict):
        record = {
            "sentence_id": self._sentence_id,
            "src_text": src_text,
            "committed_before": self._committed,
        }
        record.update(trace)
        if self._verbose:
            print(self._format_trace(record), flush=True)
        if self._trace_path is not None:
            with self._trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _format_trace(record: dict) -> str:
        lines = [
            "── TokenConsensus trace ──",
            f"sentence_id={record['sentence_id']}",
            f"src: {record['src_text']}",
            f"committed_before: {record['committed_before']!r}",
            f"delta: {record.get('delta', '')!r}",
            f"stop_reason: {record.get('stop_reason', '')}",
        ]
        futures = record.get("unique_futures", record.get("futures", []))
        if futures:
            lines.append("English futures:")
            for i, fut in enumerate(futures, 1):
                lines.append(f"  [{i}] {fut}")
        for step in record.get("steps", []):
            lines.append(
                f"step={step.get('step')} "
                f"decision={step.get('decision')} "
                f"token_id={step.get('selected_token_id')} "
                f"text={step.get('selected_text')!r} "
                f"avg_prob={step.get('selected_avg_prob', '')}"
            )
        lines.append("──")
        return "\n".join(lines)
