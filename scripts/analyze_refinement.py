"""Analyze STTR refinement effectiveness from SimulEval outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.refinement import (
    compare_triggered_sentences,
    load_directory_scores,
    load_jsonl,
    summarize_traces,
)


def format_delta(value: float) -> str:
    """Render a signed delta compactly."""
    return f"{value:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze how much STTR refinement helps over a baseline."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Directory containing the baseline SimulEval output.",
    )
    parser.add_argument(
        "--system-dir",
        type=Path,
        required=True,
        help="Directory containing the STTR or always-refine SimulEval output.",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help="Optional refinement trace JSONL. Defaults to <system-dir>/refine_trace.jsonl.",
    )
    args = parser.parse_args()

    baseline_scores = load_directory_scores(args.baseline_dir)
    system_scores = load_directory_scores(args.system_dir)

    print("Corpus-Level Comparison")
    print("-----------------------")
    print(
        f"Baseline BLEU: {baseline_scores.bleu:.2f} | "
        f"System BLEU: {system_scores.bleu:.2f} | "
        f"Delta: {format_delta(system_scores.bleu - baseline_scores.bleu)}"
    )
    print(
        f"Baseline AL:   {baseline_scores.al:.2f} | "
        f"System AL:   {system_scores.al:.2f} | "
        f"Delta: {format_delta(system_scores.al - baseline_scores.al)}"
    )

    trace_file = args.trace_file or (args.system_dir / "refine_trace.jsonl")
    if not trace_file.exists():
        print("")
        print("Trace-Level Analysis")
        print("--------------------")
        print(f"No trace file found at {trace_file}.")
        print("Corpus deltas are available, but trigger-rate and triggered-subset analysis are not.")
        return

    trace_records = load_jsonl(trace_file)
    trigger_summary = summarize_traces(trace_records)

    print("")
    print("Trace-Level Analysis")
    print("--------------------")
    print(
        f"Refined commit points: {trigger_summary.refined_commit_points}/"
        f"{trigger_summary.total_commit_points} "
        f"({trigger_summary.trigger_rate * 100:.1f}%)"
    )
    print(
        f"Triggered sentences:   {trigger_summary.triggered_sentences}/"
        f"{trigger_summary.total_sentences} "
        f"({trigger_summary.triggered_sentence_rate * 100:.1f}%)"
    )
    print(
        f"Changed outputs among refined commit points: "
        f"{trigger_summary.changed_commit_points}/"
        f"{trigger_summary.refined_commit_points} "
        f"({trigger_summary.changed_rate_among_refined * 100:.1f}%)"
    )
    if trigger_summary.read_more_commit_points:
        print(
            f"Read-more commit points: {trigger_summary.read_more_commit_points}/"
            f"{trigger_summary.total_commit_points} "
            f"({trigger_summary.read_more_rate * 100:.1f}%)"
        )

    baseline_instances = args.baseline_dir / "instances.log"
    system_instances = args.system_dir / "instances.log"
    if not baseline_instances.exists() or not system_instances.exists():
        print("")
        print("Triggered-Sentence Quality")
        print("--------------------------")
        print("instances.log is missing, so triggered-sentence quality cannot be compared yet.")
        return

    sentence_summary = compare_triggered_sentences(
        baseline_instances_path=baseline_instances,
        system_instances_path=system_instances,
        trace_records=trace_records,
    )
    print("")
    print("Triggered-Sentence Quality")
    print("--------------------------")
    print(
        f"Improved / worsened / unchanged: "
        f"{sentence_summary.improved_sentences} / "
        f"{sentence_summary.worsened_sentences} / "
        f"{sentence_summary.unchanged_sentences}"
    )
    print(
        f"Avg sentence chrF on triggered subset: "
        f"{sentence_summary.avg_baseline_chrf:.2f} -> "
        f"{sentence_summary.avg_system_chrf:.2f} "
        f"({format_delta(sentence_summary.avg_chrf_gain)})"
    )


if __name__ == "__main__":
    main()
