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
from simuleval import entrypoint
from simuleval.agents.agent import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.evaluator.instance import Instance
from transformers import MarianMTModel, MarianTokenizer

# Monkey-patch: SimulEval 1.1.0 has a bug where summarize() calls
# get_source_audio_path even for text-to-text agents.
_original_summarize = Instance.summarize


def _patched_summarize(self):
    result = {
        "index": self.index,
        "prediction": self.prediction,
        "delays": self.delays,
        "elapsed": self.elapsed,
        "prediction_length": self.prediction_length,
        "reference": self.reference,
        "source": getattr(self.dataloader, "get_source_audio_path", lambda i: "")(
            self.index
        ),
        "metric": self.metrics,
    }
    return result


Instance.summarize = _patched_summarize

# Monkey-patch: SimulEval 1.1.0 latency scorers divide by zero on empty translations
from simuleval.evaluator.scorers.latency_scorer import LatencyScorer
from statistics import mean

def _patched_scorer_call(self, instances):
    scores = []
    for ins in instances.values():
        delays = getattr(ins, self.timestamp_type)
        if not delays or ins.prediction_length == 0:
            continue
        if self.use_ref_len or ins.reference is None:
            tgt_len = ins.prediction_length
        else:
            tgt_len = len(ins.reference.split())
        src_len = ins.source_length
        if tgt_len == 0 or src_len == 0:
            continue
        scores.append(self.compute(delays, src_len, tgt_len))
    return mean(scores) if scores else 0.0

LatencyScorer.__call__ = _patched_scorer_call

# Monkey-patch: SimulEval 1.1.0 writes DataFrame to file without .to_string()
from simuleval.evaluator.evaluator import SentenceLevelEvaluator

def _patched_call(self, system):
    from tqdm.contrib.logging import logging_redirect_tqdm
    import logging

    logger = logging.getLogger("simuleval.evaluator")
    with logging_redirect_tqdm(loggers=[logger]):
        for instance in self.maybe_tqdm(self.instances.values()):
            system.reset()
            while not instance.finish_prediction:
                input_segment = instance.send_source(self.source_segment_size)
                output_segment = system.pushpop(input_segment)
                instance.receive_prediction(output_segment)
            if self.output:
                self.write_log(instance)

    results = self.results
    if self.output:
        with open(self.output / "scores", "w") as f:
            f.write(results.to_string())

    logger.info("Results:")
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
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

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
                **inputs,
                num_beams=self.args.beam_size,
                max_length=512,
            )

        translated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self._current_translation = translated.split()
        self._cached_source = source_text

        return self._current_translation
