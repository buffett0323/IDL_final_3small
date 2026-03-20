"""
STTR (Selective Test-Time Reasoning) simultaneous translation agent.

Extends wait-k with uncertainty-gated adaptation:
1. Generate a draft translation with greedy/beam decoding
2. Compute a token-level uncertainty signal
3. If uncertainty > threshold tau, either refine or read more source
4. Otherwise, use the draft as-is

Usage:
    simuleval \
        --agent agents/sttr_agent.py \
        --source data/wmt/wmt14_source.txt \
        --target data/wmt/wmt14_target.txt \
        --wait-k 5 \
        --uncertainty-threshold 2.0 \
        --output outputs/sttr_k5_tau2.0/
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
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
    add_language_args,
    build_generate_kwargs,
    load_translation_model,
)
# --- Monkey-patches (same as waitk_agent.py) ---

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
import contextlib
import json
from pathlib import Path

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

# --- STTR Agent ---


@entrypoint
class STTRAgent(TextToTextAgent):
    """
    Selective Test-Time Reasoning agent for simultaneous translation.

    At each commit point:
    1. Generate draft translation and compute uncertainty
    2. If uncertainty > tau: refine or read more source
    3. Emit the next target word when ready
    """

    def __init__(self, args):
        self._current_translation = []
        self._cached_source = None
        self._cached_uncertainty = 0.0
        self._num_refined = 0
        self._num_total = 0
        self._sentence_id = -1
        self._commit_id = 0
        self._trace_path = None

        super().__init__(args)
        self.wait_k = args.wait_k
        self.tau = args.uncertainty_threshold
        self.num_candidates = args.num_candidates
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = args.model_name
        print(f"Loading model: {model_name} on {self.device}")
        self.tokenizer, self.model, self.forced_bos_token_id = load_translation_model(
            model_name=model_name,
            device=self.device,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )

        if args.trace_refinement and getattr(args, "output", None):
            self._trace_path = Path(args.output) / "refine_trace.jsonl"

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--wait-k", type=int, default=5,
            help="Number of source words to read before starting translation",
        )
        parser.add_argument(
            "--model-name", type=str, default="Helsinki-NLP/opus-mt-en-de",
            help="HuggingFace model name for En-De translation",
        )
        add_language_args(parser)
        parser.add_argument(
            "--beam-size", type=int, default=1,
            help="Beam size for draft translation",
        )
        parser.add_argument(
            "--uncertainty-threshold", type=float, default=2.0,
            help="Entropy threshold (tau) for triggering refinement",
        )
        parser.add_argument(
            "--uncertainty-mode", type=str, default="mean",
            choices=["mean", "last", "tail3"],
            help="How to aggregate token-level entropy into one uncertainty score.",
        )
        parser.add_argument(
            "--num-candidates", type=int, default=4,
            help="Number of candidates K for refinement re-ranking",
        )
        parser.add_argument(
            "--refinement-method", type=str, default="rerank",
            choices=["rerank", "beam"],
            help="Refinement strategy: 'rerank' (multi-candidate) or 'beam' (larger beam)",
        )
        parser.add_argument(
            "--always-refine", action="store_true",
            help="Always refine (Method B upper bound). Ignores tau.",
        )
        parser.add_argument(
            "--on-uncertain", type=str, default="refine",
            choices=["refine", "read-more"],
            help="Action to take when uncertainty exceeds tau.",
        )
        parser.add_argument(
            "--trace-refinement", action="store_true",
            help="Write per-commit refinement events to output/refine_trace.jsonl.",
        )

    def reset(self):
        super().reset()
        self._current_translation = []
        self._cached_source = None
        self._cached_uncertainty = 0.0
        self._sentence_id = getattr(self, "_sentence_id", -1) + 1
        self._commit_id = 0

    def policy(self):
        src_len = len(self.states.source)
        tgt_len = len(self.states.target)

        # Wait-k: keep reading until we have k more source words than target words
        if not self.states.source_finished and src_len - tgt_len < self.wait_k:
            return ReadAction()

        # Translate the current source prefix
        source_text = " ".join(self.states.source)
        gate_result = self._translate_with_gate(source_text)
        if gate_result["gate_action"] == "read_more":
            return ReadAction()
        translation_words = gate_result["translation_words"]

        # Emit the next target word
        if tgt_len < len(translation_words):
            next_word = translation_words[tgt_len]
            finished = (
                self.states.source_finished
                and tgt_len + 1 >= len(translation_words)
            )
            return WriteAction(content=next_word, finished=finished)

        # Source finished but no more words to emit
        if self.states.source_finished:
            return WriteAction(content="", finished=True)

        # Need more source context
        return ReadAction()

    def _translate_with_gate(self, source_text):
        """Translate with uncertainty gating: draft -> gate -> maybe adapt."""
        # Cache: only re-translate if source changed
        if self._cached_source == source_text:
            return {
                "translation_words": self._current_translation,
                "gate_action": "cached",
                "uncertainty": self._cached_uncertainty,
            }

        # Step 1: Generate draft with uncertainty
        draft_words, uncertainty = self._draft_with_uncertainty(source_text)
        self._num_total += 1

        # Step 2: Gate decision
        gate_action = self._select_gate_action(uncertainty)

        if gate_action == "refine" and len(draft_words) > 0:
            self._num_refined += 1
            refined_words = self._refine(source_text)
            self._current_translation = refined_words
        elif gate_action == "read_more":
            self._current_translation = draft_words
        else:
            self._current_translation = draft_words

        changed_output = self._current_translation != draft_words
        self._maybe_trace_event(
            source_text=source_text,
            uncertainty=uncertainty,
            gate_action=gate_action,
            triggered_refine=(gate_action == "refine"),
            requested_more_read=(gate_action == "read_more"),
            changed_output=changed_output,
            draft_words=draft_words,
            final_words=self._current_translation,
        )

        self._cached_source = source_text
        self._cached_uncertainty = uncertainty

        return {
            "translation_words": self._current_translation,
            "gate_action": gate_action,
            "uncertainty": uncertainty,
        }

    def _draft_with_uncertainty(self, source_text):
        """Generate draft translation and compute mean token entropy."""
        inputs = self.tokenizer(
            source_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **build_generate_kwargs(
                    self.forced_bos_token_id,
                    **inputs,
                    num_beams=self.args.beam_size,
                    max_length=512,
                    output_scores=True, # return raw logits
                    return_dict_in_generate=True,
                )
            )

        '''
        We compute entropy: H = -sum(p * log(p)) across all 58k vocabulary tokens
        High entropy = the model spread probability across many tokens = uncertain
        Low entropy = the model is confident in one token = certain
        '''

        # Compute mean token entropy
        entropies = []
        for step_logits in outputs.scores:
            probs = F.softmax(step_logits[0], dim=-1)
            # Clamp to avoid log(0)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()
            entropies.append(entropy)

        uncertainty = self._aggregate_uncertainty(entropies)
        translated = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        return translated.split(), uncertainty

    def _aggregate_uncertainty(self, entropies):
        """Aggregate token entropies into one uncertainty score."""
        if not entropies:
            return 0.0

        if self.args.uncertainty_mode == "last":
            return entropies[-1]
        if self.args.uncertainty_mode == "tail3":
            tail = entropies[-3:]
            return sum(tail) / len(tail)
        return sum(entropies) / len(entropies)

    def _select_gate_action(self, uncertainty):
        """Choose how to adapt when the model is uncertain."""
        if self.args.always_refine:
            return "refine"

        if uncertainty <= self.tau:
            return "draft"

        if (
            self.args.on_uncertain == "read-more"
            and not self.states.source_finished
        ):
            return "read_more"

        return "refine"

    def _refine(self, source_text):
        """Refine translation via multi-candidate re-ranking."""
        inputs = self.tokenizer(
            source_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        K = self.num_candidates

        if self.args.refinement_method == "rerank":
            # Generate K candidates via diverse beam search and pick best by score
            with torch.no_grad():
                outputs = self.model.generate(
                    **build_generate_kwargs(
                        self.forced_bos_token_id,
                        **inputs,
                        num_beams=K * 2,
                        num_return_sequences=K,
                        max_length=512,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                )
            best_idx = outputs.sequences_scores.argmax()
            translated = self.tokenizer.decode(
                outputs.sequences[best_idx], skip_special_tokens=True
            )
        else:
            # "beam": simply use a larger beam
            with torch.no_grad():
                outputs = self.model.generate(
                    **build_generate_kwargs(
                        self.forced_bos_token_id,
                        **inputs,
                        num_beams=K * 2,
                        max_length=512,
                    )
                )
            translated = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

        return translated.split()

    def __del__(self):
        """Print refinement stats on cleanup."""
        if getattr(self, "_num_total", 0) > 0:
            rate = self._num_refined / self._num_total * 100
            print(
                f"\n[STTR Stats] Refined {self._num_refined}/{self._num_total} "
                f"commit points ({rate:.1f}%)"
            )

    def _maybe_trace_event(
        self,
        source_text,
        uncertainty,
        gate_action,
        triggered_refine,
        requested_more_read,
        changed_output,
        draft_words,
        final_words,
    ):
        """Optionally write one trace record for offline refinement analysis."""
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
            "triggered_refine": triggered_refine,
            "requested_more_read": requested_more_read,
            "changed_output": changed_output,
            "source_prefix": source_text,
            "draft_translation": " ".join(draft_words),
            "final_translation": " ".join(final_words),
        }
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._commit_id += 1
