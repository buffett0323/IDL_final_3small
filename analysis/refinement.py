"""Offline analysis helpers for STTR refinement experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import sacrebleu


@dataclass
class DirectoryScores:
    """Corpus-level scores for one experiment directory."""

    bleu: float
    al: float
    num_sentences: int


@dataclass
class TriggerSummary:
    """Aggregate refinement-trigger statistics."""

    total_commit_points: int
    refined_commit_points: int
    read_more_commit_points: int
    total_sentences: int
    triggered_sentences: int
    changed_commit_points: int
    changed_sentences: int

    @property
    def trigger_rate(self) -> float:
        if self.total_commit_points == 0:
            return 0.0
        return self.refined_commit_points / self.total_commit_points

    @property
    def triggered_sentence_rate(self) -> float:
        if self.total_sentences == 0:
            return 0.0
        return self.triggered_sentences / self.total_sentences

    @property
    def changed_rate_among_refined(self) -> float:
        if self.refined_commit_points == 0:
            return 0.0
        return self.changed_commit_points / self.refined_commit_points

    @property
    def read_more_rate(self) -> float:
        if self.total_commit_points == 0:
            return 0.0
        return self.read_more_commit_points / self.total_commit_points


@dataclass
class TriggeredSentenceSummary:
    """Sentence-level quality comparison on sentences that triggered refinement."""

    total_triggered_sentences: int
    improved_sentences: int
    worsened_sentences: int
    unchanged_sentences: int
    avg_baseline_chrf: float
    avg_system_chrf: float
    avg_chrf_gain: float


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL records from disk."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_from_instances(instances_path: Path) -> DirectoryScores:
    """Compute corpus BLEU and AL from an instances.log file."""
    predictions: list[str] = []
    references: list[str] = []
    als: list[float] = []

    for record in load_jsonl(instances_path):
        predictions.append(record["prediction"])
        references.append(record["reference"].strip())
        delays = record.get("delays", [])
        pred_len = record.get("prediction_length", 0)
        src_len = max(delays) if delays else 0
        als.append(compute_al(delays, src_len, pred_len))

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    avg_al = sum(als) / len(als) if als else 0.0
    return DirectoryScores(bleu=bleu, al=avg_al, num_sentences=len(predictions))


def parse_scores_file(scores_path: Path) -> DirectoryScores:
    """Read a precomputed scores file."""
    values = {}
    with scores_path.open(encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()

    return DirectoryScores(
        bleu=float(values["BLEU"]),
        al=float(values["AL"]),
        num_sentences=int(values["Sentences"]),
    )


def load_directory_scores(output_dir: Path) -> DirectoryScores:
    """Load scores from instances.log when available, otherwise use scores file."""
    instances_path = output_dir / "instances.log"
    if instances_path.exists():
        return score_from_instances(instances_path)

    scores_path = output_dir / "scores"
    if scores_path.exists():
        return parse_scores_file(scores_path)

    raise FileNotFoundError(
        f"Neither instances.log nor scores found under {output_dir}"
    )


def compute_al(delays: Iterable[int], source_length: int, target_length: int) -> float:
    """Compute Average Lagging (AL) for one sentence."""
    delays = list(delays)
    if target_length == 0 or source_length == 0 or not delays:
        return 0.0

    gamma = target_length / source_length
    al = 0.0
    tau = 0
    for t, delay in enumerate(delays):
        if delay <= source_length:
            al += delay - t / gamma
            tau = t + 1
            if delay == source_length:
                break

    if tau == 0:
        return float(source_length)
    return al / tau


def summarize_traces(trace_records: list[dict]) -> TriggerSummary:
    """Aggregate commit-level trigger statistics from STTR trace logs."""
    sentence_ids = {record["sentence_id"] for record in trace_records}
    triggered_sentence_ids = {
        record["sentence_id"]
        for record in trace_records
        if record["triggered_refine"] or record.get("requested_more_read", False)
    }
    changed_sentence_ids = {
        record["sentence_id"] for record in trace_records if record["changed_output"]
    }

    return TriggerSummary(
        total_commit_points=len(trace_records),
        refined_commit_points=sum(
            1 for record in trace_records if record["triggered_refine"]
        ),
        read_more_commit_points=sum(
            1 for record in trace_records if record.get("requested_more_read", False)
        ),
        total_sentences=len(sentence_ids),
        triggered_sentences=len(triggered_sentence_ids),
        changed_commit_points=sum(
            1 for record in trace_records if record["changed_output"]
        ),
        changed_sentences=len(changed_sentence_ids),
    )


def build_instances_by_index(instances_path: Path) -> dict[int, dict]:
    """Map sentence index to instance log record."""
    records = load_jsonl(instances_path)
    return {record["index"]: record for record in records}


def sent_chrf(prediction: str, reference: str) -> float:
    """Sentence-level chrF score for one prediction/reference pair."""
    return sacrebleu.sentence_chrf(
        hypothesis=prediction,
        references=[reference],
    ).score


def compare_triggered_sentences(
    baseline_instances_path: Path,
    system_instances_path: Path,
    trace_records: list[dict],
) -> TriggeredSentenceSummary:
    """Compare final sentence quality on the subset of sentences that triggered."""
    baseline_records = build_instances_by_index(baseline_instances_path)
    system_records = build_instances_by_index(system_instances_path)

    trace_sentence_ids = [record["sentence_id"] for record in trace_records]
    trace_id_offset = min(trace_sentence_ids) if trace_sentence_ids else 0
    triggered_sentence_ids = sorted(
        {
            record["sentence_id"] - trace_id_offset
            for record in trace_records
            if record["triggered_refine"] or record.get("requested_more_read", False)
        }
    )
    if not triggered_sentence_ids:
        return TriggeredSentenceSummary(
            total_triggered_sentences=0,
            improved_sentences=0,
            worsened_sentences=0,
            unchanged_sentences=0,
            avg_baseline_chrf=0.0,
            avg_system_chrf=0.0,
            avg_chrf_gain=0.0,
        )

    improved = 0
    worsened = 0
    unchanged = 0
    baseline_scores: list[float] = []
    system_scores: list[float] = []

    for sentence_id in triggered_sentence_ids:
        baseline = baseline_records[sentence_id]
        system = system_records[sentence_id]
        reference = system["reference"].strip()

        baseline_score = sent_chrf(baseline["prediction"], reference)
        system_score = sent_chrf(system["prediction"], reference)
        baseline_scores.append(baseline_score)
        system_scores.append(system_score)

        if system_score > baseline_score:
            improved += 1
        elif system_score < baseline_score:
            worsened += 1
        else:
            unchanged += 1

    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    avg_system = sum(system_scores) / len(system_scores)
    return TriggeredSentenceSummary(
        total_triggered_sentences=len(triggered_sentence_ids),
        improved_sentences=improved,
        worsened_sentences=worsened,
        unchanged_sentences=unchanged,
        avg_baseline_chrf=avg_baseline,
        avg_system_chrf=avg_system,
        avg_chrf_gain=avg_system - avg_baseline,
    )
