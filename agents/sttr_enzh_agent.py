"""
STTR-v2 simultaneous translation agent for EN->ZH.

Extends wait-k with:
  - NLLB-200-distilled-600M as base model
  - Character-level Chinese emission
  - Multiple uncertainty modes: mean / last / tail3 / margin
  - Uncertain -> read-more (primary) -> LCP commit (fallback)
  - Optional Qwen3 triggered rerank on hard cases

Usage:
    simuleval \
        --agent agents/sttr_enzh_agent.py \
        --source data/enzh/test_source_5.txt \
        --target data/enzh/test_target_5.txt \
        --wait-k 5 \
        --uncertainty-threshold 1.5 \
        --output outputs/enzh_sttr_smoke/
"""

import json
import os
import time
import contextlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

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

from model_utils import (
    DEFAULT_CACHE_DIR,
    add_language_args,
    build_generate_kwargs,
    load_causal_translation_model,
    load_translation_model,
    split_chinese_chars,
)
from dd_gate import compute_dd_score

# ---------------------------------------------------------------------------
# Monkey-patches (same as waitk_agent.py, needed for SimulEval 1.1.x)
# ---------------------------------------------------------------------------
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
        src_len = ins.source_length
        if src_len == 0:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _longest_common_prefix(candidates: list[list[str]]) -> list[str]:
    """Return the longest common prefix of multiple character/token lists."""
    if not candidates:
        return []
    prefix = []
    for items in zip(*candidates):
        if len(set(items)) == 1:
            prefix.append(items[0])
        else:
            break
    return prefix


def _majority_vote_at(candidates: list[list[str]], pos: int) -> str | None:
    """Return the majority character at position *pos* across candidates.

    Returns the character if a strict majority (> 50%) of candidates agree,
    otherwise None.
    """
    votes: dict[str, int] = {}
    for cand in candidates:
        if pos < len(cand):
            ch = cand[pos]
            votes[ch] = votes.get(ch, 0) + 1
    if not votes:
        return None
    best_ch, best_count = max(votes.items(), key=lambda x: x[1])
    if best_count > len(candidates) / 2:
        return best_ch
    return None


def _load_qwen(model_path: str, device: str):
    """Lazily load Qwen3 for refining.  Returns (tokenizer, model)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer as AT

    print(f"[Qwen] Loading refiner from {model_path} on {device} ...")
    tok = AT.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device,
    )
    model.eval()
    print("[Qwen] Refiner loaded.")
    return tok, model


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@entrypoint
class STTREnZhAgent(TextToTextAgent):
    """
    STTR-v2 agent: NLLB EN->ZH with read-more, LCP commit, optional Qwen rerank.
    """

    def __init__(self, args):
        # Pre-init state (reset() is called by super().__init__)
        self._current_translation: list[str] = []
        self._cached_source: str | None = None
        self._cached_uncertainty: float = 0.0

        # Stats
        self._num_total = 0
        self._num_read_more = 0
        self._num_lcp = 0
        self._num_qwen = 0
        self._lcp_lengths: list[int] = []
        self._sentence_id = -1
        self._commit_id = 0
        self._trace_path = None
        self._start_time = time.time()

        # Cache key includes source_finished so EOS triggers Qwen even when
        # source_text hasn't changed (e.g. ReadAction returns empty at end).
        self._cached_source_finished: bool = False

        super().__init__(args)
        self.wait_k = args.wait_k
        self.tau = args.uncertainty_threshold
        self.num_candidates = args.num_candidates
        self.max_extra_reads = args.max_extra_reads

        # GPU assignment
        base_device = f"cuda:{args.base_gpu}" if torch.cuda.is_available() else "cpu"
        self.device = base_device

        model_name = args.model_name
        self._causal_lm = args.causal_lm
        self._causal_prompt_template: str | None = None

        if self._causal_lm:
            print(f"[Base] Loading causal LM {model_name} on {self.device}")
            self.tokenizer, self.model, self._causal_prompt_template = (
                load_causal_translation_model(
                    model_name_or_path=model_name,
                    device=self.device,
                    cache_dir=args.cache_dir,
                )
            )
            self.forced_bos_token_id = None
        else:
            print(f"[Base] Loading seq2seq {model_name} on {self.device}")
            self.tokenizer, self.model, self.forced_bos_token_id = load_translation_model(
                model_name=model_name,
                device=self.device,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                cache_dir=args.cache_dir,
            )

        # Qwen reranker (lazy-loaded on first use)
        self._qwen_tokenizer = None
        self._qwen_model = None
        self._qwen_path = args.qwen_model_path
        self._qwen_device = f"cuda:{args.qwen_gpu}" if torch.cuda.is_available() else "cpu"
        self._qwen_enabled = bool(args.qwen_model_path)

        # Extra read tracking per sentence
        self._extra_reads_used = 0

        # Prefix-constrained continuation mode
        self._continuation = args.continuation
        self._causal_instruct = args.causal_instruct

        # DD gate state
        self._dd_enabled = args.dd_gate
        self._dd_veto = args.dd_veto          # veto mode: baseline first, DD intercepts only
        self._dd_tau = args.dd_tau
        self._dd_K = args.dd_futures_k
        self._dd_steps = args.dd_steps
        self._dd_future_mode = args.dd_future_mode      # "oracle" | "lm_sample"
        self._dd_future_words = args.dd_future_words
        self._dd_future_temperature = args.dd_future_temperature
        self._dd_cache: dict[int, dict] = {}   # prefix_len -> dd result
        self._dd_forced_reads = 0              # total READ forced by DD this sentence
        self._dd_trace_path: Path | None = None

        # Future LM for lm_sample mode: a small English causal LM that generates
        # K plausible continuations of the current observed source prefix.
        # This is loaded separately from the base MT model.
        self._future_lm = None
        self._future_lm_tokenizer = None
        if self._dd_future_mode == "lm_sample" and (self._dd_enabled or self._dd_veto):
            future_lm_path = args.dd_future_lm
            if not future_lm_path:
                raise ValueError(
                    "--dd-future-lm must be specified when --dd-future-mode=lm_sample"
                )
            # Determine device: separate GPU if specified, else same as base model
            if args.dd_future_lm_gpu is not None:
                future_lm_device = f"cuda:{args.dd_future_lm_gpu}"
            else:
                future_lm_device = self.device
            print(f"[DD-LM] Loading future LM {future_lm_path} on {future_lm_device}")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._future_lm_tokenizer = AutoTokenizer.from_pretrained(
                future_lm_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
            if self._future_lm_tokenizer.pad_token is None:
                self._future_lm_tokenizer.pad_token = self._future_lm_tokenizer.eos_token
            self._future_lm = AutoModelForCausalLM.from_pretrained(
                future_lm_path,
                cache_dir=args.cache_dir,
                torch_dtype=torch.float16,
                device_map=future_lm_device,
                trust_remote_code=True,
            )
            self._future_lm.eval()
            print(
                f"[DD-LM] Future LM loaded. Will generate K={self._dd_K} futures "
                f"of {self._dd_future_words} tokens each, T={self._dd_future_temperature}"
            )

        # Oracle source sentences (full sentences from the source file).
        # Needed for truncation futures: the streaming agent only sees words
        # 0..prefix_len-1, so without the full sentence all K futures are
        # identical -> JS=0. Loading from the source file gives genuine future
        # diversity and is the correct oracle upper-bound experiment.
        self._dd_oracle_sources: list[list[str]] | None = None
        if self._dd_enabled or self._dd_veto:
            src_file = getattr(args, "source", None)
            if src_file and Path(src_file).exists():
                with open(src_file, encoding="utf-8") as fh:
                    self._dd_oracle_sources = [
                        line.strip().split() for line in fh if line.strip()
                    ]
                print(
                    f"[DD] Loaded {len(self._dd_oracle_sources)} oracle source "
                    f"sentences from {src_file}"
                )
            else:
                print(
                    "[DD] WARNING: --source not found; all truncation futures "
                    "will be identical and JS will be 0. "
                    "Pass --source to simuleval for proper DD computation."
                )

        if args.trace_refinement and getattr(args, "output", None):
            self._trace_path = Path(args.output) / "refine_trace.jsonl"
        if (self._dd_enabled or self._dd_veto) and getattr(args, "output", None):
            dd_tp = Path(args.output) / "dd_trace.jsonl"
            # Always start with a fresh trace file so stale records from a
            # previous partial/wrong run do not corrupt the new one.
            if dd_tp.exists():
                dd_tp.unlink()
            self._dd_trace_path = dd_tp

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--wait-k", type=int, default=5,
            help="Number of source words to read before starting translation",
        )
        parser.add_argument(
            "--model-name", type=str,
            default="facebook/nllb-200-distilled-600M",
            help="HuggingFace model name for base translation",
        )
        add_language_args(parser)
        parser.add_argument(
            "--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
            help="Local cache directory for model downloads",
        )
        parser.add_argument(
            "--base-gpu", type=int, default=0,
            help="GPU index for base NLLB model",
        )
        parser.add_argument(
            "--beam-size", type=int, default=1,
            help="Beam size for draft translation",
        )
        parser.add_argument(
            "--uncertainty-threshold", type=float, default=3.0,
            help="Threshold (tau) for triggering adaptation",
        )
        parser.add_argument(
            "--uncertainty-mode", type=str, default="tail3",
            choices=["mean", "last", "tail3", "margin", "seq_logprob"],
            help=(
                "How to aggregate token-level uncertainty. "
                "'mean'/'last'/'tail3': token entropy variants. "
                "'margin': 1 - top1/top2 prob gap. "
                "'seq_logprob': -mean log-prob of chosen tokens (sequence-level confidence)."
            ),
        )
        parser.add_argument(
            "--num-candidates", type=int, default=4,
            help="Number of candidates K for LCP commit",
        )
        parser.add_argument(
            "--max-extra-reads", type=int, default=3,
            help="Max additional source words to read before falling back to LCP",
        )
        parser.add_argument(
            "--always-refine", action="store_true",
            help="Always use LCP (Method B upper bound). Ignores tau.",
        )
        parser.add_argument(
            "--trace-refinement", action="store_true",
            help="Write per-commit trace to output/refine_trace.jsonl",
        )
        # Qwen reranker options
        parser.add_argument(
            "--qwen-model-path", type=str, default="",
            help="Path to local Qwen3 model for triggered reranking (empty = disabled)",
        )
        parser.add_argument(
            "--qwen-gpu", type=int, default=1,
            help="GPU index for Qwen reranker",
        )
        parser.add_argument(
            "--qwen-skip-ratio", type=float, default=1.0,
            help="Skip Qwen when committed/draft ratio >= this (1.0 = never skip)",
        )
        parser.add_argument(
            "--causal-lm", action="store_true",
            help=(
                "Use a causal (decoder-only) LM as the base translator instead of a "
                "seq2seq model (e.g. Qwen3-4B-Base instead of NLLB)."
            ),
        )
        parser.add_argument(
            "--causal-instruct", action="store_true",
            help=(
                "When --causal-lm is set, treat the model as a chat/instruct model "
                "(e.g. Qwen3-30B-A3B-Instruct) instead of a base model. "
                "Uses apply_chat_template() for prompting with a system message that "
                "requests Simplified Chinese, and appends the committed target to the "
                "assistant turn for prefix-constrained continuation. "
                "Also sets enable_thinking=False to skip the <think> block. "
                "Requires --continuation for the streaming path."
            ),
        )
        parser.add_argument(
            "--qwen-mode", type=str, default="prefix",
            choices=["prefix", "rerank", "logprob_rerank"],
            help=(
                "'prefix': feed committed chars as assistant prefix, generate only the "
                "continuation (no full re-translation). "
                "'rerank': generate K base-model beam candidates, ask Qwen to pick the best "
                "via a verbal choice. "
                "'logprob_rerank': like rerank but Qwen scores candidates by log-probability "
                "(more principled than verbal selection)."
            ),
        )
        # ── DD gate ────────────────────────────────────────────────────────────
        parser.add_argument(
            "--dd-gate", action="store_true",
            help=(
                "Enable Distribution Divergence (DD) full gate: DD score is computed "
                "before every potential commit decision (post wait-k). Forces READ "
                "whenever avg_js_firstN > --dd-tau, replacing the baseline commit."
            ),
        )
        parser.add_argument(
            "--dd-veto", action="store_true",
            help=(
                "Enable DD veto mode (soft gate): the baseline policy (wait-k + "
                "uncertainty gate) decides first. Only when the baseline would commit "
                "does DD get to veto — if avg_js_firstN > --dd-tau the commit is "
                "converted to READ. DD is never invoked when baseline already READs."
            ),
        )
        parser.add_argument(
            "--dd-tau", type=float, default=0.05,
            help="DD gate threshold. READ if avg_js_firstN > dd_tau. Default 0.05.",
        )
        parser.add_argument(
            "--dd-futures-k", type=int, default=4,
            help="Number of English futures to sample for DD computation.",
        )
        parser.add_argument(
            "--dd-steps", type=int, default=3,
            help=(
                "Number of Chinese decoding steps to average JS over. "
                "3 is a good default (avg_js_first3). "
                "Use 1 for speed (avg_js_first1, single-step only)."
            ),
        )
        parser.add_argument(
            "--dd-future-mode", type=str, default="oracle",
            choices=["oracle", "lm_sample"],
            help=(
                "How to generate K English futures for DD computation.\n"
                "'oracle' (default): deterministically reveals 1..K more words "
                "from the FULL source sentence (upper-bound experiment).\n"
                "'lm_sample': uses --dd-future-lm to sample K diverse continuations "
                "of the current observed prefix — no oracle needed. "
                "This is the realistic inference-time mode."
            ),
        )
        parser.add_argument(
            "--dd-future-lm", type=str, default=None,
            help=(
                "Path or HuggingFace model ID for the English future-sampling LM. "
                "Required when --dd-future-mode=lm_sample. "
                "Recommended: a small causal LM such as Qwen/Qwen3-4B (base). "
                "This LM generates English continuations only; translation is "
                "still done by the base MT model."
            ),
        )
        parser.add_argument(
            "--dd-future-lm-gpu", type=int, default=None,
            help=(
                "GPU index for the future LM. Defaults to same GPU as base model. "
                "Set to a different value to split across GPUs if memory is tight."
            ),
        )
        parser.add_argument(
            "--dd-future-words", type=int, default=15,
            help="Number of tokens to generate per sampled English future. Default 15.",
        )
        parser.add_argument(
            "--dd-future-temperature", type=float, default=0.9,
            help=(
                "Sampling temperature for future LM. Higher = more diverse futures. "
                "Default 0.9.  Use 1.0 for maximum diversity."
            ),
        )
        parser.add_argument(
            "--continuation", action="store_true",
            help=(
                "Enable prefix-constrained continuation mode.\n"
                "Instead of re-translating the full source prefix and indexing "
                "into the draft by position (translation[tgt_len]), this mode "
                "explicitly conditions the decoder on the already-committed target "
                "prefix via decoder_input_ids, then generates only what comes next.\n"
                "This eliminates translation-hypothesis inconsistency: the committed "
                "prefix is always respected and the model cannot 'forget' what was "
                "already written.  Works only with seq2seq (NLLB) base models."
            ),
        )

    def reset(self):
        super().reset()
        self._current_translation = []
        self._cached_source = None
        self._cached_uncertainty = 0.0
        self._cached_source_finished = False
        self._sentence_id = getattr(self, "_sentence_id", -1) + 1
        self._commit_id = 0
        self._extra_reads_used = 0
        self._dd_cache = {}
        self._dd_forced_reads = 0
        # Continuation mode: cache encoder output for the current source prefix
        # so we avoid re-encoding when writing multiple chars under the same prefix.
        self._cont_enc_cache: dict[str, object] = {}
        self._cont_attn_cache: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def policy(self):
        src_len = len(self.states.source)
        tgt_len = len(self.states.target)

        # Wait-k: keep reading until we have k more source words than target words
        if not self.states.source_finished and src_len - tgt_len < self.wait_k:
            return ReadAction()

        # Prefix-constrained continuation mode: bypass full re-translation.
        if self._continuation:
            return self._continuation_policy(src_len, tgt_len)

        # DD full gate: force READ before even calling the baseline uncertainty gate.
        # DD is the primary decision-maker post wait-k; baseline entropy gate is
        # bypassed when DD says READ.  Applied only during streaming (not at EOS).
        if self._dd_enabled and not self.states.source_finished:
            dd_result = self._dd_cached_score(src_len)
            dd_score = dd_result["avg_js_firstN"]
            dd_decision = "COMMIT" if dd_score <= self._dd_tau else "READ"
            self._maybe_trace_dd(src_len, tgt_len, dd_result, dd_decision)
            if dd_decision == "READ":
                self._dd_forced_reads += 1
                return ReadAction()

        # Translate the current source prefix (runs baseline uncertainty gate)
        source_text = " ".join(self.states.source)
        gate_result = self._translate_with_gate(source_text)

        if gate_result["gate_action"] == "read_more":
            return ReadAction()

        # DD veto: baseline already decided to commit — DD may intercept only here.
        # Unlike DD full gate, veto respects the baseline's own uncertainty gate:
        # if baseline already said READ (read_more), DD is never consulted.
        # DD only fires when baseline would commit AND DD detects high divergence.
        if self._dd_veto and not self.states.source_finished:
            dd_result = self._dd_cached_score(src_len)
            dd_score = dd_result["avg_js_firstN"]
            dd_decision = "COMMIT" if dd_score <= self._dd_tau else "READ"
            self._maybe_trace_dd(src_len, tgt_len, dd_result, dd_decision)
            if dd_decision == "READ":
                self._dd_forced_reads += 1
                return ReadAction()

        translation_units = gate_result["translation_units"]

        # Emit the next target unit (character)
        if tgt_len < len(translation_units):
            next_unit = translation_units[tgt_len]
            finished = (
                self.states.source_finished
                and tgt_len + 1 >= len(translation_units)
            )
            return WriteAction(content=next_unit, finished=finished)

        # Source finished but no more units to emit
        if self.states.source_finished:
            return WriteAction(content="", finished=True)

        return ReadAction()

    # ------------------------------------------------------------------
    # Prefix-constrained continuation (NLLB seq2seq + Causal LM)
    # ------------------------------------------------------------------

    def _continuation_policy(self, src_len: int, tgt_len: int):
        """Policy when --continuation is enabled.

        At each step we ask the model: "given the source prefix and the target
        tokens we have already committed, what is the NEXT Chinese character?"

        Unlike re-translation (which ignores the committed prefix and uses
        translation[tgt_len]), continuation always produces output that is
        consistent with what was already emitted — it cannot contradict itself.

        Dispatches to:
          - _causal_prefix_continuation()  when --causal-lm is set (e.g. Qwen)
          - _nllb_prefix_continuation()    for seq2seq (NLLB)

        Safety cap: if committed is already 3x longer than source words, assume
        EOS and stop to avoid infinite loops on pathological inputs.
        """
        source_text = " ".join(self.states.source)
        committed = list(self.states.target)

        # Safety cap — avoid runaway generation
        if tgt_len >= src_len * 3 + 5:
            return WriteAction(content="", finished=True)

        if self._causal_lm:
            next_char, is_eos = self._causal_prefix_continuation(source_text, committed)
        else:
            next_char, is_eos = self._nllb_prefix_continuation(source_text, committed)

        if is_eos or not next_char:
            if self.states.source_finished:
                return WriteAction(content="", finished=True)
            # EOS on partial source: translation done for this prefix,
            # but more source is coming — read it
            return ReadAction()

        # Emit the next character
        # finished = True only when source is done AND the model hit EOS
        finished = self.states.source_finished and is_eos
        self._num_total += 1
        return WriteAction(content=next_char, finished=finished)

    # Few-shot template for causal-LM prefix continuation.
    # Explicitly requests Simplified Chinese and ends with "Chinese: " so the
    # model continues the translation from where <committed_target> leaves off.
    # ── Few-shot template for base (non-instruct) causal LM continuation ────────
    _CONT_FEW_SHOT = (
        "Translate English to Simplified Chinese (简体中文).\n\n"
        "English: Macedonians go to polls in referendum on changing country's name.\n"
        "Chinese: 马其顿人就更改国名举行公投。\n\n"
        "English: Orlando Bloom and Miranda Kerr still love each other.\n"
        "Chinese: 奥兰多·布鲁姆和米兰达·可儿仍然彼此相爱。\n\n"
        "English: {source}\n"
        "Chinese: "
    )

    # ── System prompt for instruct-model continuation ─────────────────────────
    _INSTRUCT_SYSTEM = (
        "You are a professional translator. "
        "Translate English text to Simplified Chinese (简体中文). "
        "Output ONLY the Chinese translation. "
        "Do NOT add explanations, notes, or extra text."
    )

    def _build_instruct_continuation_prompt(
        self, source_text: str, committed_text: str
    ) -> str:
        """Build an instruct-model prompt for prefix-constrained continuation.

        The committed Chinese text is appended to the assistant turn so the
        model sees it as "already written" and continues naturally:

            <|im_start|>system
            You are a professional translator...
            <|im_end|>
            <|im_start|>user
            Translate to Simplified Chinese: "AMs are apparently..."
            <|im_end|>
            <|im_start|>assistant
            AMs似乎在提出              ← model continues HERE
        """
        messages = [
            {"role": "system", "content": self._INSTRUCT_SYSTEM},
            {"role": "user",   "content": f"Translate to Simplified Chinese:\n{source_text}"},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # disable Qwen3 think block
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Append committed target — model continues the assistant turn from here
        return prompt + committed_text

    def _causal_prefix_continuation(
        self, source_text: str, committed: list[str]
    ) -> tuple[str, bool]:
        """Causal-LM prefix-constrained continuation.

        Supports two prompt styles via --causal-instruct flag:

        BASE MODEL (--causal-lm, few-shot):
            Translate English to Simplified Chinese ...
            English: <source>
            Chinese: <committed>  ← model continues

        INSTRUCT MODEL (--causal-lm --causal-instruct, chat template):
            <system>You are a professional translator...</system>
            <user>Translate: <source></user>
            <assistant><committed>  ← model continues

        The instruct path produces higher-quality output: the model can follow
        the explicit "Simplified Chinese" instruction, and the committed text
        sits naturally inside the assistant turn.

        Stop conditions (both paths):
          - EOS token                        → translation complete
          - Newline (base) / im_end (inst.)  → turn boundary hit
          - First 8 tokens contain no CJK    → treat as EOS

        Returns (next_char, is_eos).
        """
        committed_text = "".join(committed)

        # ── Build prompt ──────────────────────────────────────────────────────
        if self._causal_instruct:
            prompt = self._build_instruct_continuation_prompt(source_text, committed_text)
            # im_end token marks the end of the assistant turn
            im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            extra_stop = im_end_ids[:1] if im_end_ids else []
        else:
            prompt = self._CONT_FEW_SHOT.format(source=source_text) + committed_text
            extra_stop = []

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # Stop at EOS or newline/im_end
        newline_id = self.tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_id[:1] if newline_id else []
        stop_ids = list({self.tokenizer.eos_token_id} | set(newline_id) | set(extra_stop))

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_ids,
                repetition_penalty=1.05,
            )

        new_ids = out[0][input_len:]
        if len(new_ids) == 0:
            return "", True
        if new_ids[0].item() in stop_ids:
            return "", True

        continuation_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        continuation_text = continuation_text.split("\n")[0].strip()

        chars = split_chinese_chars(continuation_text)
        if not chars:
            return "", True

        return chars[0], False

    def _nllb_prefix_continuation(
        self, source_text: str, committed: list[str]
    ) -> tuple[str, bool]:
        """Force-decode NLLB with the committed target prefix, get next char.

        Decoder sequence fed as context (decoder_input_ids):
            [decoder_start_id, zho_Hans_id, *committed_token_ids]

        The model generates tokens AFTER this prefix.  The first decoded
        character of the continuation becomes the next committed character.

        Returns:
            (next_char, is_eos):
                next_char — the next Chinese character to emit ('' if none)
                is_eos    — True if the model produced EOS before any char
        """
        if self._causal_lm:
            raise RuntimeError(
                "--continuation with seq2seq path called but --causal-lm is set. "
                "This is a bug; _causal_prefix_continuation should have been used."
            )

        committed_text = "".join(committed)

        # ── Encoder: cache by source_text to avoid redundant encoding ──────────
        if source_text not in self._cont_enc_cache:
            raw_inputs = self.tokenizer(
                source_text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                encoder = self.model.get_encoder()
                enc_out = encoder(
                    input_ids=raw_inputs["input_ids"],
                    attention_mask=raw_inputs["attention_mask"],
                    return_dict=True,
                )
            # Keep only the latest source prefix to bound memory usage
            self._cont_enc_cache = {source_text: enc_out}
            self._cont_attn_cache = {source_text: raw_inputs["attention_mask"]}

        enc_out = self._cont_enc_cache[source_text]
        attn_mask = self._cont_attn_cache[source_text]

        # ── Decoder prefix: [decoder_start, lang_token, *committed_tokens] ─────
        if committed_text:
            committed_ids = self.tokenizer(
                committed_text, add_special_tokens=False
            ).input_ids
        else:
            committed_ids = []

        decoder_start = self.model.config.decoder_start_token_id
        decoder_prefix_ids = [decoder_start, self.forced_bos_token_id] + committed_ids
        decoder_prefix = torch.tensor([decoder_prefix_ids]).to(self.device)

        # ── Generate continuation ────────────────────────────────────────────────
        with torch.no_grad():
            out = self.model.generate(
                encoder_outputs=enc_out,
                attention_mask=attn_mask,
                decoder_input_ids=decoder_prefix,
                forced_bos_token_id=None,   # lang token already in decoder_prefix
                max_new_tokens=8,           # enough for 1-2 CJK characters
                num_beams=1,
                do_sample=False,
            )

        prefix_len = decoder_prefix.shape[1]
        new_ids = out[0][prefix_len:]

        eos_id = self.tokenizer.eos_token_id
        if len(new_ids) == 0 or new_ids[0].item() == eos_id:
            return "", True

        continuation_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        chars = split_chinese_chars(continuation_text)
        if not chars:
            return "", True

        return chars[0], False

    # ------------------------------------------------------------------
    # Core translation + gating
    # ------------------------------------------------------------------

    def _translate_with_gate(self, source_text: str) -> dict:
        """Translate with uncertainty gating.

        Cache key is (source_text, source_finished) so that when source_finished
        transitions to True (e.g. ReadAction returns empty at EOS), we re-run
        the gate and trigger Qwen even when source_text hasn't changed.
        """
        src_finished_now = self.states.source_finished
        if (self._cached_source == source_text
                and self._cached_source_finished == src_finished_now):
            return {
                "translation_units": self._current_translation,
                "gate_action": "cached",
                "uncertainty": self._cached_uncertainty,
            }

        # Step 1: draft with uncertainty
        draft_units, uncertainty = self._draft_with_uncertainty(source_text)
        self._num_total += 1

        # Step 2: gate decision
        gate_action = self._select_gate_action(uncertainty)

        tgt_len = len(self.states.target)

        if gate_action == "read_more":
            self._num_read_more += 1
            self._extra_reads_used += 1
            self._current_translation = draft_units
        elif gate_action == "qwen_refine":
            self._num_qwen += 1
            committed = list(self.states.target) if self.states.target else []
            refined = self._qwen_suffix_refine(source_text, committed, draft_units)
            self._current_translation = refined
        elif gate_action == "lcp":
            self._num_lcp += 1
            merged = self._lcp_merge(source_text, draft_units, tgt_len)
            self._current_translation = merged
        else:
            # draft (confident)
            self._current_translation = draft_units

        self._maybe_trace_event(
            source_text=source_text,
            uncertainty=uncertainty,
            gate_action=gate_action,
            draft_units=draft_units,
            final_units=self._current_translation,
        )

        self._cached_source = source_text
        self._cached_source_finished = src_finished_now
        self._cached_uncertainty = uncertainty

        return {
            "translation_units": self._current_translation,
            "gate_action": gate_action,
            "uncertainty": uncertainty,
        }

    def _draft_with_uncertainty(self, source_text: str):
        """Generate draft translation and compute uncertainty.

        Supports both seq2seq (NLLB) and causal LM (Qwen3-4B-Base) base models.
        """
        if self._causal_lm:
            return self._draft_causal(source_text)
        return self._draft_seq2seq(source_text)

    def _draft_seq2seq(self, source_text: str):
        """Draft generation for seq2seq models (NLLB etc.)."""
        inputs = self.tokenizer(
            source_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        gen_kwargs = build_generate_kwargs(
            self.forced_bos_token_id,
            **inputs,
            num_beams=self.args.beam_size,
            max_length=512,
            output_scores=True,
            return_dict_in_generate=True,
        )

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        entropies, margins, chosen_log_probs = [], [], []
        for step_idx, step_logits in enumerate(outputs.scores):
            probs = F.softmax(step_logits[0], dim=-1)
            log_probs_dist = F.log_softmax(step_logits[0], dim=-1)
            entropy = -(probs * log_probs_dist).sum().item()
            entropies.append(entropy)
            top2 = probs.topk(2).values
            margins.append((top2[0] - top2[1]).item())
            chosen_id = outputs.sequences[0][step_idx + 1].item()
            chosen_log_probs.append(log_probs_dist[chosen_id].item())

        uncertainty = self._aggregate_uncertainty(entropies, margins, chosen_log_probs)
        translated = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return split_chinese_chars(translated), uncertainty

    def _draft_causal(self, source_text: str):
        """Draft generation for causal / decoder-only LMs (e.g. Qwen3-4B-Base).

        We build a few-shot prompt, generate the Chinese translation, then compute
        per-token uncertainty exactly as in the seq2seq path.
        """
        prompt = self._causal_prompt_template.format(source=source_text)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Uncertainty comes from the generated (Chinese) tokens only
        entropies, margins, chosen_log_probs = [], [], []
        for step_idx, step_logits in enumerate(outputs.scores):
            probs = F.softmax(step_logits[0], dim=-1)
            log_probs_dist = F.log_softmax(step_logits[0], dim=-1)
            entropy = -(probs * log_probs_dist).sum().item()
            entropies.append(entropy)
            top2 = probs.topk(2).values
            margins.append((top2[0] - top2[1]).item())
            chosen_id = outputs.sequences[0][prompt_len + step_idx].item()
            chosen_log_probs.append(log_probs_dist[chosen_id].item())

        uncertainty = self._aggregate_uncertainty(entropies, margins, chosen_log_probs)

        # Decode only new tokens (the Chinese translation)
        new_ids = outputs.sequences[0][prompt_len:]
        translated = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Stop at double newline — the model may start the next few-shot example
        translated = translated.split("\n")[0].strip()

        return split_chinese_chars(translated), uncertainty

    def _aggregate_uncertainty(
        self,
        entropies: list[float],
        margins: list[float],
        chosen_log_probs: list[float] | None = None,
    ) -> float:
        """Aggregate per-token signals into one uncertainty score."""
        mode = self.args.uncertainty_mode
        if not entropies:
            return 0.0

        if mode == "last":
            return entropies[-1]
        if mode == "tail3":
            tail = entropies[-3:]
            return sum(tail) / len(tail)
        if mode == "margin":
            # 1 - margin: higher = more uncertain
            tail = margins[-3:]
            avg_margin = sum(tail) / len(tail)
            return 1.0 - avg_margin
        if mode == "seq_logprob":
            # uncertainty = negative mean log-prob of chosen tokens
            # higher value = less confident = more uncertain
            if not chosen_log_probs:
                return sum(entropies) / len(entropies)
            return -sum(chosen_log_probs) / len(chosen_log_probs)
        # default: mean entropy
        return sum(entropies) / len(entropies)

    def _select_gate_action(self, uncertainty: float) -> str:
        """Choose adaptation strategy.

        Key policy:
          - At EOS (source_finished=True): ALWAYS trigger Qwen if enabled,
            regardless of uncertainty. This is the core of the always-refine
            design (use --uncertainty-threshold 999 to disable streaming gates).
          - During streaming with tau < 999: use read-more up to max_extra_reads,
            then draft.
          - LCP is a fallback when source is finished but Qwen is disabled.

        With --always-refine: streaming always drafts (no read-more), EOS always
        triggers Qwen (if enabled) or LCP (if disabled).
        """
        if self.args.always_refine:
            if self.states.source_finished:
                return "qwen_refine" if self._qwen_enabled else "lcp"
            # always draft during streaming — no read-more, minimal latency
            return "draft"

        # EOS: unconditional Qwen (no uncertainty gate)
        if self.states.source_finished and self._qwen_enabled:
            return "qwen_refine"

        if uncertainty <= self.tau:
            return "draft"

        # Uncertain during streaming: prefer read-more
        if not self.states.source_finished:
            if self._extra_reads_used < self.max_extra_reads:
                return "read_more"
            return "draft"

        # Source finished, Qwen disabled
        return "lcp"

    # ------------------------------------------------------------------
    # DD gate helpers
    # ------------------------------------------------------------------

    def _dd_cached_score(self, prefix_len: int) -> dict:
        """Return cached DD score for current source prefix length.

        Recomputed only when prefix_len changes (new source word arrived).

        Uses oracle_source_words (full sentence loaded at startup) so that
        truncation futures are genuinely diverse.  Falls back to observed
        prefix if oracle is unavailable (will give JS=0 for most steps).
        """
        if prefix_len not in self._dd_cache:
            # _sentence_id is incremented once during super().__init__() and once
            # per sentence start by SimulEval.  So during sentence 1 it is 1,
            # during sentence 2 it is 2, etc.  Subtract 1 for 0-based indexing.
            oracle_idx = self._sentence_id - 1
            if (self._dd_oracle_sources is not None
                    and 0 <= oracle_idx < len(self._dd_oracle_sources)):
                oracle_words = self._dd_oracle_sources[oracle_idx]
            else:
                # Fallback: only observed prefix — futures will be identical
                oracle_words = list(self.states.source)

            self._dd_cache[prefix_len] = compute_dd_score(
                model=self.model,
                tokenizer=self.tokenizer,
                oracle_source_words=oracle_words,
                prefix_len=prefix_len,
                device=self.device,
                causal_lm=self._causal_lm,
                forced_bos_token_id=self.forced_bos_token_id,
                prompt_template=self._causal_prompt_template,
                K=self._dd_K,
                n_steps=self._dd_steps,
                future_mode=self._dd_future_mode,
                future_lm=self._future_lm,
                future_lm_tokenizer=self._future_lm_tokenizer,
                future_words=self._dd_future_words,
                future_temperature=self._dd_future_temperature,
            )
        return self._dd_cache[prefix_len]

    def _maybe_trace_dd(
        self,
        src_len: int,
        tgt_len: int,
        dd_result: dict,
        decision: str,
    ) -> None:
        """Write one DD gate decision to dd_trace.jsonl (if trace enabled).

        Record includes:
          - sentence_id, src_len, tgt_len
          - src_prefix: the observed English prefix text
          - avg_js_first1 / avg_js_first3 / avg_js_firstN: DD scores
          - per_step_js: per-step breakdown
          - dd_tau, decision (COMMIT / READ)
          - baseline_decision: always COMMIT (DD gate is post-wait-k; baseline
            would proceed to translate at this point)
          - futures: the K English futures used for DD computation
        """
        if self._dd_trace_path is None:
            return
        src_prefix = " ".join(self.states.source)
        record = {
            "sentence_id": self._sentence_id,
            "src_len": src_len,
            "tgt_len": tgt_len,
            "src_prefix": src_prefix,
            "avg_js_first1": dd_result.get("avg_js_first1", 0.0),
            "avg_js_first3": dd_result.get("avg_js_first3", 0.0),
            "avg_js_first5": dd_result.get("avg_js_first5", 0.0),
            "avg_js_firstN": dd_result.get("avg_js_firstN", 0.0),
            "per_step_js": dd_result.get("per_step_js", []),
            "K": dd_result.get("K", self._dd_K),
            "n_steps": dd_result.get("n_steps", self._dd_steps),
            "dd_tau": self._dd_tau,
            "decision": decision,
            "baseline_decision": "COMMIT",  # baseline always commits post-wait-k
            "futures": dd_result.get("futures", []),
            "future_mode": dd_result.get("future_mode", self._dd_future_mode),
        }
        with self._dd_trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # LCP commit: generate K candidates, commit their common prefix
    # ------------------------------------------------------------------

    def _generate_candidates(self, source_text: str) -> list[list[str]]:
        """Generate K candidate translations, each split into char units.

        Works for both seq2seq (beam search over full sequence) and causal LM
        (beam search starting from the prompt prefix).
        """
        K = self.num_candidates

        if self._causal_lm:
            prompt = self._causal_prompt_template.format(source=source_text)
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    num_beams=K * 2,
                    num_return_sequences=K,
                    max_new_tokens=96,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            candidates = []
            for seq in outputs:
                text = self.tokenizer.decode(
                    seq[prompt_len:], skip_special_tokens=True
                ).strip().split("\n")[0].strip()
                candidates.append(split_chinese_chars(text))
        else:
            inputs = self.tokenizer(
                source_text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **build_generate_kwargs(
                        self.forced_bos_token_id,
                        **inputs,
                        num_beams=K * 2,
                        num_return_sequences=K,
                        max_length=512,
                        do_sample=False,
                    )
                )

            candidates = []
            for seq in outputs:
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                candidates.append(split_chinese_chars(text))

        return candidates

    def _lcp_merge(
        self, source_text: str, draft_units: list[str], tgt_len: int
    ) -> list[str]:
        """Position-aware LCP merge: build a stable translation using candidates.

        Strategy:
        1. Keep everything already committed (positions < tgt_len) from draft.
        2. From tgt_len onward, extend using LCP (all candidates agree).
        3. When LCP ends, continue with majority vote (>50% agree).
        4. When majority vote also fails, fall back to draft.
        5. Never return fewer units than draft (no truncation).

        If Qwen is enabled and source is finished but no candidate agreement
        at tgt_len, trigger Qwen rerank.
        """
        candidates = self._generate_candidates(source_text)
        if not candidates:
            return draft_units

        # Build merged translation
        merged = list(draft_units[:tgt_len])  # already committed prefix

        # Phase 1: extend with strict LCP from tgt_len onward
        pos = tgt_len
        lcp_len = 0
        while True:
            chars = set()
            all_have = True
            for cand in candidates:
                if pos < len(cand):
                    chars.add(cand[pos])
                else:
                    all_have = False
                    break
            if all_have and len(chars) == 1:
                merged.append(chars.pop())
                lcp_len += 1
                pos += 1
            else:
                break

        # Phase 2: extend with majority vote
        mv_len = 0
        while True:
            ch = _majority_vote_at(candidates, pos)
            if ch is not None:
                merged.append(ch)
                mv_len += 1
                pos += 1
            else:
                break

        self._lcp_lengths.append(lcp_len)

        # Never truncate: if merged is shorter than draft, use draft tail
        if len(merged) < len(draft_units):
            merged.extend(draft_units[len(merged):])

        return merged

    # ------------------------------------------------------------------
    # Qwen triggered refinement (two modes)
    # ------------------------------------------------------------------

    def _ensure_qwen_loaded(self):
        if self._qwen_model is None:
            self._qwen_tokenizer, self._qwen_model = _load_qwen(
                self._qwen_path, self._qwen_device
            )

    def _qwen_suffix_refine(
        self, source_text: str, committed: list[str], draft_units: list[str]
    ) -> list[str]:
        """Dispatch to the selected Qwen mode."""
        self._ensure_qwen_loaded()

        mode = getattr(self.args, "qwen_mode", "prefix")
        try:
            if mode == "rerank":
                return self._qwen_rerank(source_text, committed, draft_units)
            elif mode == "logprob_rerank":
                return self._qwen_logprob_rerank(source_text, committed, draft_units)
            else:
                return self._qwen_prefix_continuation(source_text, committed, draft_units)
        except Exception as e:
            print(f"[Qwen] Refine failed ({mode}): {e}", flush=True)
            return draft_units

    def _qwen_prefix_continuation(
        self, source_text: str, committed: list[str], draft_units: list[str]
    ) -> list[str]:
        """Prefix-constrained Qwen continuation.

        Feed committed_text as the START of the assistant turn so Qwen
        cannot output those tokens again — it only generates what comes after.

        Flow:
          1. Build chat-template prompt ending with <|im_start|>assistant\\n.
          2. Append committed_text → model sees it as "already written".
          3. Generate max_new_tokens from that position; decode ONLY new ids.
          4. Strip the committed prefix from the decoded text if it reappears
             (token-boundary misalignment).
          5. Return committed_units + clean_continuation_units.
        """
        committed_text = "".join(committed)

        user_msg = (
            "Translate the following English sentence into Chinese. "
            "Output ONLY the Chinese translation, nothing else.\n\n"
            f"English: {source_text}"
        )
        messages = [{"role": "user", "content": user_msg}]

        # Build prompt ending with <|im_start|>assistant\n, then append committed.
        # enable_thinking=False suppresses the <think>...</think> block in Qwen3.
        try:
            prompt_text = self._qwen_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self._qwen_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompt_with_prefix = prompt_text + committed_text

        inputs = self._qwen_tokenizer(
            prompt_with_prefix, return_tensors="pt",
            truncation=True, max_length=2048,
        ).to(self._qwen_device)

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self._qwen_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=self._qwen_tokenizer.eos_token_id,
            )

        # Decode only new tokens after the prompt+prefix
        continuation_raw = self._qwen_tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True,
        ).strip()

        # Stop at end-of-turn markers if they appear
        for stop in ("<|im_end|>", "<|endoftext|>", "\n\n"):
            if stop in continuation_raw:
                continuation_raw = continuation_raw[:continuation_raw.index(stop)]
        continuation_raw = continuation_raw.strip()

        # Repetition fix: strip committed prefix if model re-generates it
        if committed_text and continuation_raw.startswith(committed_text):
            continuation_raw = continuation_raw[len(committed_text):]

        if not continuation_raw and not committed:
            return draft_units

        continuation_units = split_chinese_chars(continuation_raw)
        result = list(committed) + continuation_units
        return result if len(result) >= len(draft_units) else draft_units

    def _qwen_rerank(
        self, source_text: str, committed: list[str], draft_units: list[str]
    ) -> list[str]:
        """Use Qwen to score NLLB beam candidates and pick the best.

        This avoids prefix-alignment issues entirely: we let NLLB generate K
        candidates (all starting with the committed prefix), then ask Qwen to
        pick the best one.  The winning candidate replaces the draft.

        Qwen scores each candidate as a quality judge (no translation needed).
        """
        candidates = self._generate_candidates(source_text)
        if not candidates:
            return draft_units

        # Filter candidates that match the committed prefix
        valid = []
        for c in candidates:
            if c[:len(committed)] == list(committed) or len(committed) == 0:
                valid.append(c)
        if not valid:
            valid = candidates

        # If only one valid candidate, just return it
        if len(valid) == 1:
            result = valid[0]
            return result if len(result) >= len(draft_units) else draft_units

        # Ask Qwen to pick the best translation
        candidate_texts = ["".join(c) for c in valid]
        options_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(candidate_texts))
        user_msg = (
            f"You are evaluating Chinese translations of an English sentence.\n"
            f"Pick the BEST translation (most accurate, natural Chinese).\n\n"
            f"English: {source_text}\n\n"
            f"Candidates:\n{options_str}\n\n"
            f"Reply with ONLY the number of the best translation (e.g. '2')."
        )
        messages = [{"role": "user", "content": user_msg}]

        raw = self._qwen_generate_text(messages)
        # Parse the chosen index
        import re as _re
        m = _re.search(r"\d+", raw)
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(valid):
                chosen = valid[idx]
                if len(chosen) >= len(draft_units):
                    return chosen

        return draft_units

    def _qwen_logprob_rerank(
        self, source_text: str, committed: list[str], draft_units: list[str]
    ) -> list[str]:
        """Score NLLB/base-model beam candidates using Qwen log-probabilities.

        Unlike verbal rerank (which asks Qwen to pick a number), this method
        computes P(candidate | source) using Qwen's own log-probs — more
        principled and doesn't rely on instruction following for scoring.

        Steps:
          1. Generate K beam candidates from the base model.
          2. Filter candidates that are consistent with committed prefix.
          3. For each candidate, score it with Qwen: average log-prob of the
             Chinese tokens conditioned on the source.
          4. Return the highest-scoring candidate.
        """
        candidates = self._generate_candidates(source_text)
        if not candidates:
            return draft_units

        # Filter to candidates compatible with committed prefix
        n = len(committed)
        valid = [c for c in candidates if c[:n] == list(committed) or n == 0]
        if not valid:
            valid = candidates

        if len(valid) == 1:
            result = valid[0]
            return result if len(result) >= len(draft_units) else draft_units

        # Score each candidate with Qwen log-prob
        best_candidate = draft_units
        best_score = float("-inf")

        score_prompt_prefix = (
            f"Translate to Chinese:\nEnglish: {source_text}\nChinese: "
        )
        prefix_ids = self._qwen_tokenizer(
            score_prompt_prefix, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._qwen_device)
        prefix_len = prefix_ids.shape[1]

        for cand in valid:
            cand_text = "".join(cand)
            full_prompt = score_prompt_prefix + cand_text
            inputs = self._qwen_tokenizer(
                full_prompt, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(self._qwen_device)

            with torch.no_grad():
                out = self._qwen_model(**inputs)

            # Log-probs at each position (shift by 1 for causal LM)
            logits = out.logits[0, :-1]          # (seq_len-1, vocab)
            log_probs = F.log_softmax(logits, dim=-1)
            target_ids = inputs.input_ids[0, 1:]  # (seq_len-1,)
            token_lps = log_probs[range(len(target_ids)), target_ids]

            # Average log-prob of candidate tokens only (after the prompt prefix)
            cand_lps = token_lps[prefix_len - 1:]
            if len(cand_lps) == 0:
                continue
            score = cand_lps.mean().item()

            if score > best_score:
                best_score = score
                best_candidate = cand

        result = list(best_candidate)
        return result if len(result) >= len(draft_units) else draft_units

    def _qwen_generate_text(self, messages: list[dict]) -> str:
        """Run Qwen generation and return raw text response."""
        try:
            text = self._qwen_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self._qwen_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self._qwen_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self._qwen_device)

        with torch.no_grad():
            out = self._qwen_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
            )
        return self._qwen_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def _qwen_generate(self, messages: list[dict]) -> list[str]:
        """Run Qwen generation and return split Chinese char units."""
        return split_chinese_chars(self._qwen_generate_text(messages))

    # ------------------------------------------------------------------
    # Tracing & stats
    # ------------------------------------------------------------------

    def _maybe_trace_event(self, source_text, uncertainty, gate_action,
                           draft_units, final_units):
        if self._trace_path is None:
            return

        record = {
            "sentence_id": self._sentence_id,
            "commit_id": self._commit_id,
            "source_length": len(self.states.source),
            "target_length_before_write": len(self.states.target),
            "uncertainty": uncertainty,
            "threshold": self.tau,
            "uncertainty_mode": self.args.uncertainty_mode,
            "gate_action": gate_action,
            "source_prefix": source_text,
            "draft_translation": "".join(draft_units),
            "final_translation": "".join(final_units),
        }
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._commit_id += 1

    def __del__(self):
        total = getattr(self, "_num_total", 0)
        if total == 0:
            return
        elapsed = time.time() - self._start_time
        read_more_rate = self._num_read_more / total * 100
        lcp_rate = self._num_lcp / total * 100
        qwen_rate = self._num_qwen / total * 100
        avg_lcp_len = (
            sum(self._lcp_lengths) / len(self._lcp_lengths)
            if self._lcp_lengths else 0.0
        )
        print(
            f"\n{'='*60}\n"
            f"[STTR-v2 Stats]\n"
            f"  Total commit points : {total}\n"
            f"  Read-more triggered : {self._num_read_more} ({read_more_rate:.1f}%)\n"
            f"  LCP commit triggered: {self._num_lcp} ({lcp_rate:.1f}%)\n"
            f"  Avg LCP length      : {avg_lcp_len:.1f} chars\n"
            f"  Qwen rerank calls   : {self._num_qwen} ({qwen_rate:.1f}%)\n"
            f"  Wall time           : {elapsed:.1f}s\n"
            f"{'='*60}"
        )
