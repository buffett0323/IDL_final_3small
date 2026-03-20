"""
Wait-k simultaneous translation agent for SimulEval.

Reads k source words before starting to translate, then alternates:
read 1 source word, write 1 target word.

Usage:
    simuleval \
        --agent agents/waitk_agent.py \
        --source data/wmt/wmt14_source.txt \
        --target data/wmt/wmt14_target.txt \
        --wait-k 5 \
        --output outputs/baseline_k5/
"""

import torch
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

# Monkey-patch: SimulEval 1.1.0 has a bug where summarize() calls
# get_source_audio_path even for text-to-text agents.
_original_summarize = Instance.summarize


def _patched_summarize(self):
    result = _original_summarize(self)
    result["metric"] = self.metrics
    return result


Instance.summarize = _patched_summarize

# Monkey-patch: SimulEval 1.1.0 latency scorers divide by zero on empty translations
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

# Monkey-patch: SimulEval 1.1.0 writes DataFrame to file without .to_string()
from simuleval.evaluator.evaluator import SentenceLevelEvaluator
from simuleval.data.dataloader.dataloader import IterableDataloader
import contextlib
import json

def _patched_call(self, system):
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


SentenceLevelEvaluator.__call__ = _patched_call


@entrypoint
class WaitKAgent(TextToTextAgent):
    """Wait-k simultaneous translation agent using Helsinki-NLP/opus-mt-en-de."""

    def __init__(self, args):
        super().__init__(args)
        self.wait_k = args.wait_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = args.model_name
        print(f"Loading model: {model_name} on {self.device}")
        self.tokenizer, self.model, self.forced_bos_token_id = load_translation_model(
            model_name=model_name,
            device=self.device,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )

        self._current_translation = []

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
            "--beam-size", type=int, default=4,
            help="Beam size for translation",
        )

    def reset(self):
        super().reset()
        self._current_translation = []

    def policy(self):
        src_len = len(self.states.source)
        tgt_len = len(self.states.target)

        # Wait-k: keep reading until we have k more source words than target words
        if not self.states.source_finished and src_len - tgt_len < self.wait_k:
            return ReadAction()

        # Translate the current source prefix
        source_text = " ".join(self.states.source)
        translation_words = self._translate(source_text)

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

    def _translate(self, source_text):
        """Translate source text and return list of target words."""
        # Cache: only re-translate if source changed
        if (
            hasattr(self, "_cached_source")
            and self._cached_source == source_text
        ):
            return self._current_translation

        inputs = self.tokenizer(
            source_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **build_generate_kwargs(
                    self.forced_bos_token_id,
                    **inputs,
                    num_beams=self.args.beam_size,
                    max_length=512,
                )
            )

        translated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self._current_translation = translated.split()
        self._cached_source = source_text

        return self._current_translation
