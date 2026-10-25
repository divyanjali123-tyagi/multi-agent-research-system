"""Unit tests for routing (selector function), approval parsing and termination."""

import pytest
from autogen_agentchat.messages import TextMessage, ToolCallSummaryMessage

from research_agents.agents import (
    GAP_ANALYST,
    INSIGHT_EXTRACTOR,
    PAPER_FINDER,
    REPORT_COMPILER,
    TOPIC_REFINER,
    USER,
)
from research_agents.workflow import WorkflowRouter, is_approval, user_exit_requested


def msg(source: str, content: str = "x") -> TextMessage:
    return TextMessage(source=source, content=content)


def tool_summary(content: str = "Retrieved 3 papers") -> ToolCallSummaryMessage:
    return ToolCallSummaryMessage(source=PAPER_FINDER, content=content, tool_calls=[], results=[])


TASK = msg("user", "Research topic: graph neural networks")


@pytest.mark.parametrize(
    "text", ["", "APPROVE", "approve", "yes", "Y", "ok", "LGTM!", "looks good", "I approve"]
)
def test_is_approval_true(text):
    assert is_approval(text)


@pytest.mark.parametrize("text", ["focus on chemistry instead", "no", "change the scope", "not yet"])
def test_is_approval_false(text):
    assert not is_approval(text)


def test_first_turn_goes_to_topic_refiner():
    assert WorkflowRouter()([TASK]) == TOPIC_REFINER


def test_refiner_hands_off_to_human_checkpoint():
    assert WorkflowRouter()([TASK, msg(TOPIC_REFINER)]) == USER


def test_approval_moves_to_paper_finder():
    history = [TASK, msg(TOPIC_REFINER), msg(USER, "APPROVE")]
    assert WorkflowRouter()(history) == PAPER_FINDER


def test_feedback_loops_back_to_refiner():
    history = [TASK, msg(TOPIC_REFINER), msg(USER, "focus on molecules")]
    assert WorkflowRouter()(history) == TOPIC_REFINER


def test_revision_budget_exhausted_proceeds():
    router = WorkflowRouter(max_revisions=1)
    history = [
        TASK,
        msg(TOPIC_REFINER),
        msg(USER, "narrower"),
        msg(TOPIC_REFINER),
        msg(USER, "even narrower"),
    ]
    assert router(history) == PAPER_FINDER


def test_sequential_pipeline_after_search():
    router = WorkflowRouter()
    base = [TASK, msg(TOPIC_REFINER), msg(USER, "yes"), tool_summary()]
    assert router(base) == INSIGHT_EXTRACTOR
    assert router(base + [msg(INSIGHT_EXTRACTOR)]) == REPORT_COMPILER
    assert router(base + [msg(INSIGHT_EXTRACTOR), msg(REPORT_COMPILER)]) == GAP_ANALYST


def test_paper_finder_retries_if_it_skipped_the_tool():
    router = WorkflowRouter(max_search_attempts=2)
    history = [TASK, msg(TOPIC_REFINER), msg(USER, "yes"), msg(PAPER_FINDER, "I think...")]
    assert router(history) == PAPER_FINDER
    history.append(msg(PAPER_FINDER, "still no tool"))
    assert router(history) == INSIGHT_EXTRACTOR


def test_user_exit_detection():
    assert user_exit_requested([msg(USER, "exit")])
    assert user_exit_requested([msg(USER, "  QUIT ")])
    assert not user_exit_requested([msg(USER, "exit strategies for startups")])
    assert not user_exit_requested([msg(TOPIC_REFINER, "exit")])
