"""Tests for STTR refinement analysis utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analysis.refinement import compare_triggered_sentences, summarize_traces


class TriggerSummaryTest(unittest.TestCase):
    def test_summarize_traces_counts_commit_and_sentence_stats(self) -> None:
        records = [
            {
                "sentence_id": 0,
                "triggered_refine": False,
                "requested_more_read": False,
                "changed_output": False,
            },
            {
                "sentence_id": 0,
                "triggered_refine": True,
                "requested_more_read": False,
                "changed_output": True,
            },
            {
                "sentence_id": 1,
                "triggered_refine": True,
                "requested_more_read": False,
                "changed_output": False,
            },
        ]

        summary = summarize_traces(records)

        self.assertEqual(summary.total_commit_points, 3)
        self.assertEqual(summary.refined_commit_points, 2)
        self.assertEqual(summary.read_more_commit_points, 0)
        self.assertEqual(summary.total_sentences, 2)
        self.assertEqual(summary.triggered_sentences, 2)
        self.assertEqual(summary.changed_commit_points, 1)
        self.assertEqual(summary.changed_sentences, 1)
        self.assertAlmostEqual(summary.trigger_rate, 2 / 3)
        self.assertAlmostEqual(summary.read_more_rate, 0.0)
        self.assertAlmostEqual(summary.triggered_sentence_rate, 1.0)
        self.assertAlmostEqual(summary.changed_rate_among_refined, 0.5)


class TriggeredSentenceComparisonTest(unittest.TestCase):
    def test_compare_triggered_sentences_uses_triggered_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            baseline_path = tmp_path / "baseline.jsonl"
            system_path = tmp_path / "system.jsonl"

            baseline_path.write_text(
                "\n".join(
                    [
                        (
                            '{"index": 0, "prediction": "hello there", '
                            '"reference": "hello world\\n"}'
                        ),
                        (
                            '{"index": 1, "prediction": "good night", '
                            '"reference": "good morning\\n"}'
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            system_path.write_text(
                "\n".join(
                    [
                        (
                            '{"index": 0, "prediction": "hello world", '
                            '"reference": "hello world\\n"}'
                        ),
                        (
                            '{"index": 1, "prediction": "bad night", '
                            '"reference": "good morning\\n"}'
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            trace_records = [
                {
                    "sentence_id": 0,
                    "triggered_refine": True,
                    "requested_more_read": False,
                    "changed_output": True,
                },
                {
                    "sentence_id": 1,
                    "triggered_refine": False,
                    "requested_more_read": False,
                    "changed_output": False,
                },
            ]

            summary = compare_triggered_sentences(
                baseline_instances_path=baseline_path,
                system_instances_path=system_path,
                trace_records=trace_records,
            )

            self.assertEqual(summary.total_triggered_sentences, 1)
            self.assertEqual(summary.improved_sentences, 1)
            self.assertEqual(summary.worsened_sentences, 0)
            self.assertEqual(summary.unchanged_sentences, 0)
            self.assertGreater(summary.avg_system_chrf, summary.avg_baseline_chrf)
            self.assertGreater(summary.avg_chrf_gain, 0.0)


if __name__ == "__main__":
    unittest.main()
