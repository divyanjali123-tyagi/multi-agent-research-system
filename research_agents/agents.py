"""Factory functions for the specialised agents in the research team."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Union

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient

from . import prompts

# Agent names are used by the selector function to route the conversation.
TOPIC_REFINER = "topic_refiner"
USER = "user"
PAPER_FINDER = "paper_finder"
INSIGHT_EXTRACTOR = "insight_extractor"
REPORT_COMPILER = "report_compiler"
GAP_ANALYST = "gap_analyst"

WORKFLOW_ORDER = [
    TOPIC_REFINER,
    USER,
    PAPER_FINDER,
    INSIGHT_EXTRACTOR,
    REPORT_COMPILER,
    GAP_ANALYST,
]

InputFunc = Union[
    Callable[[str], str],
    Callable[[str, CancellationToken | None], Awaitable[str]],
]
ClientFactory = Callable[[str], ChatCompletionClient]


@dataclass
class ResearchAgents:
    topic_refiner: AssistantAgent
    user: UserProxyAgent
    paper_finder: AssistantAgent
    insight_extractor: AssistantAgent
    report_compiler: AssistantAgent
    gap_analyst: AssistantAgent

    def as_list(self) -> list:
        return [
            self.topic_refiner,
            self.user,
            self.paper_finder,
            self.insight_extractor,
            self.report_compiler,
            self.gap_analyst,
        ]


def build_agents(
    client_factory: ClientFactory,
    search_tool: Callable[..., Awaitable[str]],
    input_func: InputFunc | None = None,
) -> ResearchAgents:
    """Create every agent. ``client_factory(role)`` returns the model client for a role."""

    topic_refiner = AssistantAgent(
        name=TOPIC_REFINER,
        model_client=client_factory(TOPIC_REFINER),
        description="Refines a broad research interest into a focused direction with arXiv queries.",
        system_message=prompts.TOPIC_REFINER,
    )

    user = UserProxyAgent(
        name=USER,
        description="The human researcher who approves or revises the research direction.",
        input_func=input_func,
    )

    paper_finder = AssistantAgent(
        name=PAPER_FINDER,
        model_client=client_factory(PAPER_FINDER),
        description="Searches arXiv for papers matching the approved research direction.",
        system_message=prompts.PAPER_FINDER,
        tools=[search_tool],
        # Return the raw tool output as the agent's message so downstream agents
        # see the full paper list (no lossy LLM summary in between).
        reflect_on_tool_use=False,
        tool_call_summary_format="{result}",
    )

    insight_extractor = AssistantAgent(
        name=INSIGHT_EXTRACTOR,
        model_client=client_factory(INSIGHT_EXTRACTOR),
        description="Extracts problems, methods, findings and limitations from retrieved papers.",
        system_message=prompts.INSIGHT_EXTRACTOR,
    )

    report_compiler = AssistantAgent(
        name=REPORT_COMPILER,
        model_client=client_factory(REPORT_COMPILER),
        description="Compiles extracted insights into a structured literature review.",
        system_message=prompts.REPORT_COMPILER,
    )

    gap_analyst = AssistantAgent(
        name=GAP_ANALYST,
        model_client=client_factory(GAP_ANALYST),
        description="Identifies research gaps and future directions, then ends the run.",
        system_message=prompts.GAP_ANALYST,
    )

    return ResearchAgents(
        topic_refiner=topic_refiner,
        user=user,
        paper_finder=paper_finder,
        insight_extractor=insight_extractor,
        report_compiler=report_compiler,
        gap_analyst=gap_analyst,
    )
