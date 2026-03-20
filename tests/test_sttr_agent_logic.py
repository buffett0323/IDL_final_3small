"""Unit tests for STTR gate logic that do not require model loading."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agents.sttr_agent import STTRAgent


class STTRAgentLogicTest(unittest.TestCase):
    def make_agent(self, uncertainty_mode="mean", on_uncertain="refine", source_finished=False):
        agent = STTRAgent.__new__(STTRAgent)
        agent.args = SimpleNamespace(
            uncertainty_mode=uncertainty_mode,
            on_uncertain=on_uncertain,
            always_refine=False,
        )
        agent.tau = 2.0
        agent.states = SimpleNamespace(source_finished=source_finished)
        return agent

    def test_aggregate_uncertainty_mean(self) -> None:
        agent = self.make_agent(uncertainty_mode="mean")
        self.assertAlmostEqual(agent._aggregate_uncertainty([1.0, 2.0, 4.0]), 7.0 / 3.0)

    def test_aggregate_uncertainty_last(self) -> None:
        agent = self.make_agent(uncertainty_mode="last")
        self.assertEqual(agent._aggregate_uncertainty([1.0, 2.0, 4.0]), 4.0)

    def test_aggregate_uncertainty_tail3_uses_recent_tokens(self) -> None:
        agent = self.make_agent(uncertainty_mode="tail3")
        self.assertAlmostEqual(agent._aggregate_uncertainty([1.0, 2.0, 4.0, 8.0]), (2.0 + 4.0 + 8.0) / 3.0)

    def test_select_gate_action_refine_when_uncertain(self) -> None:
        agent = self.make_agent(on_uncertain="refine", source_finished=False)
        self.assertEqual(agent._select_gate_action(2.5), "refine")

    def test_select_gate_action_read_more_when_uncertain_and_source_not_finished(self) -> None:
        agent = self.make_agent(on_uncertain="read-more", source_finished=False)
        self.assertEqual(agent._select_gate_action(2.5), "read_more")

    def test_select_gate_action_falls_back_to_refine_at_end_of_source(self) -> None:
        agent = self.make_agent(on_uncertain="read-more", source_finished=True)
        self.assertEqual(agent._select_gate_action(2.5), "refine")

    def test_select_gate_action_keeps_draft_when_confident(self) -> None:
        agent = self.make_agent(on_uncertain="read-more", source_finished=False)
        self.assertEqual(agent._select_gate_action(1.5), "draft")


if __name__ == "__main__":
    unittest.main()
