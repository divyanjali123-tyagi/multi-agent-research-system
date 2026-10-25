"""Workflow orchestration: selector function, termination conditions and team builder.

The team is an AutoGen ``SelectorGroupChat`` whose speaker selection is fully
deterministic. A custom ``selector_func`` inspects the conversation (the
*workflow state*) and routes to the next agent:

    task ─► topic_refiner ─► user (approval checkpoint)
                 ▲               │
                 └── revise ◄────┤
                                 └── approve ─► paper_finder ─► insight_extractor
                                                 ─► report_compiler ─► gap_analyst ─► TERMINATE
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.conditions import (
    FunctionalTermination,
    MaxMessageTermination,
    TextMentionTermination,
    TimeoutTermination,
)
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.models import ChatCompletionClient

from . import prompts
from .agents import (
    GAP_ANALYST,
    INSIGHT_EXTRACTOR,
    PAPER_FINDER,
    REPORT_COMPILER,
    TOPIC_REFINER,
    USER,
    ResearchAgents,
)

Messages = Sequence[BaseAgentEvent | BaseChatMessage]

_APPROVAL_PATTERN = re.compile(
    r"^\s*(approve[d]?|yes|y|ok|okay|lgtm|proceed|go|looks good|sounds good)\b[\s.!]*",
    re.IGNORECASE,
)
EXIT_WORDS = {"exit", "quit", "stop", "abort"}


def is_approval(text: str) -> bool:
    """A reply counts as approval if it is empty (Enter) or starts with an approval word."""
    if text is None:
        return False
    stripped = text.strip()
    if stripped == "":
        return True
    if prompts.APPROVE_KEYWORD.lower() in stripped.lower().split():
        return True
    return bool(_APPROVAL_PATTERN.match(stripped))


def _chat_messages(messages: Messages) -> list[BaseChatMessage]:
    return [m for m in messages if isinstance(m, BaseChatMessage)]


def _text(message: BaseChatMessage) -> str:
    return message.to_text() if hasattr(message, "to_text") else str(getattr(message, "content", ""))


@dataclass
class WorkflowRouter:
    """Callable selector function that routes by workflow state.

    The router is stateless with respect to the conversation: everything is
    derived from the message history, so it behaves correctly after resets or
    when the team is resumed.
    """

    max_revisions: int = 3
    max_search_attempts: int = 2

    def count_revisions(self, chat: list[BaseChatMessage]) -> int:
        # chat[0] is the initial task (also sourced "user"), so skip it.
        return sum(1 for m in chat[1:] if m.source == USER and not is_approval(_text(m)))

    def count_search_attempts(self, chat: list[BaseChatMessage]) -> int:
        return sum(1 for m in chat if m.source == PAPER_FINDER)

    def __call__(self, messages: Messages) -> str | None:
        chat = _chat_messages(messages)

        # Only the task so far -> start by refining the topic.
        if len(chat) <= 1:
            return TOPIC_REFINER

        last = chat[-1]
        source = last.source

        if source == TOPIC_REFINER:
            return USER  # human-in-the-loop checkpoint

        if source == USER:
            if is_approval(_text(last)):
                return PAPER_FINDER
            if self.count_revisions(chat) > self.max_revisions:
                # Revision budget exhausted: continue with the latest proposal.
                return PAPER_FINDER
            return TOPIC_REFINER

        if source == PAPER_FINDER:
            used_tool = isinstance(last, ToolCallSummaryMessage)
            if not used_tool and self.count_search_attempts(chat) < self.max_search_attempts:
                return PAPER_FINDER  # model answered without searching - retry once
            return INSIGHT_EXTRACTOR

        if source == INSIGHT_EXTRACTOR:
            return REPORT_COMPILER

        if source == REPORT_COMPILER:
            return GAP_ANALYST

        # gap_analyst normally ends the run via TERMINATE; if it did not,
        # give it one more turn and let MaxMessageTermination stop runaway loops.
        return GAP_ANALYST


def user_exit_requested(messages: Messages) -> bool:
    """Stop immediately if the human types exit/quit/stop/abort at a checkpoint."""
    for m in messages:
        if isinstance(m, BaseChatMessage) and m.source == USER and _text(m).strip().lower() in EXIT_WORDS:
            return True
    return False


def build_termination(max_messages: int, timeout_seconds: float) -> TerminationCondition:
    return (
        TextMentionTermination(prompts.TERMINATE_KEYWORD, sources=[GAP_ANALYST])
        | FunctionalTermination(user_exit_requested)
        | MaxMessageTermination(max_messages)
        | TimeoutTermination(timeout_seconds)
    )


def build_team(
    agents: ResearchAgents,
    selector_model_client: ChatCompletionClient,
    *,
    max_messages: int = 40,
    timeout_seconds: float = 900,
    max_revisions: int = 3,
) -> SelectorGroupChat:
    """Assemble the SelectorGroupChat with our deterministic router.

    ``selector_model_client`` is required by AutoGen but is only consulted if
    the selector function returns ``None`` (it never does in normal operation).
    """
    return SelectorGroupChat(
        participants=agents.as_list(),
        model_client=selector_model_client,
        selector_func=WorkflowRouter(max_revisions=max_revisions),
        termination_condition=build_termination(max_messages, timeout_seconds),
        allow_repeated_speaker=True,
        name="research_team",
        description="Multi-agent academic research pipeline.",
    )
