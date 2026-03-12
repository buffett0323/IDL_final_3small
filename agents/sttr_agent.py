"""
STTR (Selective Test-Time Reasoning) simultaneous translation agent.

Extends wait-k with uncertainty-gated refinement:
1. Generate draft translation with greedy/beam decoding
2. Compute token-level entropy as uncertainty signal
3. If uncertainty > threshold tau, refine via multi-candidate re-ranking
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
from simuleval import entrypoint
from simuleval.agents.agent import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.evaluator.instance import Instance
from transformers import MarianMTModel, MarianTokenizer

# --- Monkey-patches (same as waitk_agent.py) ---

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

from simuleval.evaluator.evaluator import SentenceLevelEvaluator

def _patched_eval_call(self, system):
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

SentenceLevelEvaluator.__call__ = _patched_eval_call

# --- STTR Agent ---


@entrypoint
class STTRAgent(TextToTextAgent):
    """
    Selective Test-Time Reasoning agent for simultaneous translation.

    At each commit point:
    1. Generate draft translation and compute uncertainty (mean token entropy)
    2. If uncertainty > tau: refine via multi-candidate re-ranking
    3. Emit the next target word
    """

    def __init__(self, args):
        super().__init__(args)
        self.wait_k = args.wait_k
        self.tau = args.uncertainty_threshold
        self.num_candidates = args.num_candidates
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = args.model_name
        print(f"Loading model: {model_name} on {self.device}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        self._current_translation = []
        self._cached_source = None
        self._cached_uncertainty = 0.0

        # Stats tracking
        self._num_refined = 0
        self._num_total = 0

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
            "--beam-size", type=int, default=1,
            help="Beam size for draft translation",
        )
        parser.add_argument(
            "--uncertainty-threshold", type=float, default=2.0,
            help="Entropy threshold (tau) for triggering refinement",
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

    def reset(self):
        super().reset()
        self._current_translation = []
        self._cached_source = None
        self._cached_uncertainty = 0.0

    def policy(self):
        src_len = len(self.states.source)
        tgt_len = len(self.states.target)

        # Wait-k: keep reading until we have k more source words than target words
        if not self.states.source_finished and src_len - tgt_len < self.wait_k:
            return ReadAction()

        # Translate the current source prefix
        source_text = " ".join(self.states.source)
        translation_words = self._translate_with_gate(source_text)

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
        """Translate with uncertainty gating: draft -> gate -> maybe refine."""
        # Cache: only re-translate if source changed
        if self._cached_source == source_text:
            return self._current_translation

        # Step 1: Generate draft with uncertainty
        draft_words, uncertainty = self._draft_with_uncertainty(source_text)
        self._num_total += 1

        # Step 2: Gate decision
        should_refine = self.args.always_refine or (uncertainty > self.tau)

        if should_refine and len(draft_words) > 0:
            self._num_refined += 1
            refined_words = self._refine(source_text)
            self._current_translation = refined_words
        else:
            self._current_translation = draft_words

        self._cached_source = source_text
        self._cached_uncertainty = uncertainty

        return self._current_translation

    def _draft_with_uncertainty(self, source_text):
        """Generate draft translation and compute mean token entropy."""
        inputs = self.tokenizer(
            source_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=self.args.beam_size,
                max_length=512,
                output_scores=True, # return raw logits
                return_dict_in_generate=True,
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

        uncertainty = sum(entropies) / len(entropies) if entropies else 0.0
        translated = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        return translated.split(), uncertainty

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
                    **inputs,
                    num_beams=K * 2,
                    num_return_sequences=K,
                    max_length=512,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            best_idx = outputs.sequences_scores.argmax()
            translated = self.tokenizer.decode(
                outputs.sequences[best_idx], skip_special_tokens=True
            )
        else:
            # "beam": simply use a larger beam
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    num_beams=K * 2,
                    max_length=512,
                )
            translated = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

        return translated.split()

    def __del__(self):
        """Print refinement stats on cleanup."""
        if self._num_total > 0:
            rate = self._num_refined / self._num_total * 100
            print(
                f"\n[STTR Stats] Refined {self._num_refined}/{self._num_total} "
                f"commit points ({rate:.1f}%)"
            )
