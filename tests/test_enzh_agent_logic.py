"""Unit tests for STTR-v2 EN->ZH agent logic (no model loading)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agents.model_utils import split_chinese_chars, segment_chinese_reference
from agents.sttr_enzh_agent import STTREnZhAgent, _longest_common_prefix, _majority_vote_at


class ChineseSplitTest(unittest.TestCase):
    def test_pure_chinese(self):
        self.assertEqual(split_chinese_chars("你好世界"), ["你", "好", "世", "界"])

    def test_mixed_chinese_english(self):
        self.assertEqual(
            split_chinese_chars("我喜欢Python编程"),
            ["我", "喜", "欢", "Python", "编", "程"],
        )

    def test_chinese_with_spaces(self):
        self.assertEqual(split_chinese_chars("你 好"), ["你", "好"])

    def test_numbers(self):
        self.assertEqual(split_chinese_chars("第3名"), ["第", "3", "名"])

    def test_empty(self):
        self.assertEqual(split_chinese_chars(""), [])

    def test_segment_reference(self):
        self.assertEqual(segment_chinese_reference("你好世界"), "你 好 世 界")

    def test_segment_reference_mixed(self):
        self.assertEqual(
            segment_chinese_reference("我喜欢Python"),
            "我 喜 欢 Python",
        )


class LCPTest(unittest.TestCase):
    def test_basic_lcp(self):
        self.assertEqual(
            _longest_common_prefix([["你", "好", "世"], ["你", "好", "人"]]),
            ["你", "好"],
        )

    def test_no_common_prefix(self):
        self.assertEqual(
            _longest_common_prefix([["我"], ["你"]]),
            [],
        )

    def test_identical(self):
        self.assertEqual(
            _longest_common_prefix([["你", "好"], ["你", "好"]]),
            ["你", "好"],
        )

    def test_single_candidate(self):
        self.assertEqual(
            _longest_common_prefix([["你", "好", "世", "界"]]),
            ["你", "好", "世", "界"],
        )

    def test_empty(self):
        self.assertEqual(_longest_common_prefix([]), [])


class MajorityVoteTest(unittest.TestCase):
    def test_majority_exists(self):
        cands = [["你", "好"], ["你", "坏"], ["你", "好"]]
        self.assertEqual(_majority_vote_at(cands, 0), "你")
        self.assertEqual(_majority_vote_at(cands, 1), "好")

    def test_no_majority(self):
        cands = [["你"], ["我"], ["他"], ["她"]]
        self.assertIsNone(_majority_vote_at(cands, 0))

    def test_position_beyond_candidate(self):
        cands = [["你", "好"], ["你"]]
        # pos 1: only 1 of 2 has it, "好" has 1/2 = 50%, not strict majority
        self.assertIsNone(_majority_vote_at(cands, 1))

    def test_3_of_4_majority(self):
        cands = [["A"], ["A"], ["A"], ["B"]]
        self.assertEqual(_majority_vote_at(cands, 0), "A")

    def test_2_of_4_no_majority(self):
        cands = [["A"], ["A"], ["B"], ["B"]]
        self.assertIsNone(_majority_vote_at(cands, 0))


class GateLogicTest(unittest.TestCase):
    def _make_agent(self, uncertainty_mode="tail3", source_finished=False,
                    extra_reads_used=0, max_extra_reads=3, always_refine=False,
                    qwen_enabled=False):
        agent = STTREnZhAgent.__new__(STTREnZhAgent)
        agent.args = SimpleNamespace(
            uncertainty_mode=uncertainty_mode,
            always_refine=always_refine,
        )
        agent.tau = 1.5
        agent.max_extra_reads = max_extra_reads
        agent.states = SimpleNamespace(source_finished=source_finished)
        agent._extra_reads_used = extra_reads_used
        agent._qwen_enabled = qwen_enabled
        return agent

    def test_confident_returns_draft(self):
        agent = self._make_agent()
        self.assertEqual(agent._select_gate_action(0.5), "draft")

    def test_uncertain_returns_read_more(self):
        agent = self._make_agent()
        self.assertEqual(agent._select_gate_action(2.0), "read_more")

    def test_uncertain_reads_exhausted_source_not_finished_returns_draft(self):
        """When reads exhausted but source not done, return draft (don't block)."""
        agent = self._make_agent(extra_reads_used=3, source_finished=False)
        self.assertEqual(agent._select_gate_action(2.0), "draft")

    def test_uncertain_source_finished_returns_lcp_no_qwen(self):
        agent = self._make_agent(source_finished=True)
        self.assertEqual(agent._select_gate_action(2.0), "lcp")

    def test_uncertain_source_finished_returns_qwen_when_enabled(self):
        agent = self._make_agent(source_finished=True, qwen_enabled=True)
        self.assertEqual(agent._select_gate_action(2.0), "qwen_refine")

    def test_qwen_not_triggered_before_source_finished(self):
        """Qwen must NOT trigger while source is still streaming."""
        agent = self._make_agent(extra_reads_used=3, qwen_enabled=True,
                                 source_finished=False)
        self.assertNotEqual(agent._select_gate_action(2.0), "qwen_refine")

    def test_always_refine_returns_qwen_when_source_finished(self):
        agent = self._make_agent(always_refine=True, qwen_enabled=True,
                                 source_finished=True)
        self.assertEqual(agent._select_gate_action(0.1), "qwen_refine")

    def test_always_refine_returns_lcp_no_qwen(self):
        agent = self._make_agent(always_refine=True, source_finished=True)
        self.assertEqual(agent._select_gate_action(0.1), "lcp")

    def test_aggregate_mean(self):
        agent = self._make_agent(uncertainty_mode="mean")
        result = agent._aggregate_uncertainty([1.0, 2.0, 3.0], [0.9, 0.8, 0.7])
        self.assertAlmostEqual(result, 2.0)

    def test_aggregate_last(self):
        agent = self._make_agent(uncertainty_mode="last")
        result = agent._aggregate_uncertainty([1.0, 2.0, 5.0], [0.9, 0.8, 0.7])
        self.assertEqual(result, 5.0)

    def test_aggregate_tail3(self):
        agent = self._make_agent(uncertainty_mode="tail3")
        result = agent._aggregate_uncertainty([0.5, 1.0, 2.0, 3.0], [0.9, 0.8, 0.7, 0.6])
        self.assertAlmostEqual(result, 2.0)

    def test_aggregate_margin(self):
        agent = self._make_agent(uncertainty_mode="margin")
        # margins = [0.9, 0.8, 0.7], tail3 avg = 0.8, uncertainty = 1 - 0.8 = 0.2
        result = agent._aggregate_uncertainty([1.0, 2.0, 3.0], [0.9, 0.8, 0.7])
        self.assertAlmostEqual(result, 0.2)


if __name__ == "__main__":
    unittest.main()
