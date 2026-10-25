"""High-level entry point that wires settings, agents, team and output together."""

from __future__ import annotations

import asyncio
import logging

from autogen_agentchat.base import TaskResult
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

from .agents import InputFunc, build_agents
from .config import Settings
from .report import ResearchOutput, build_output, save_output
from .tools.arxiv_tool import ArxivSearchTool, Fetcher, PaperStore
from .workflow import build_team

logger = logging.getLogger(__name__)

CHECKPOINT_BANNER = (
    "\n" + "=" * 70 + "\n"
    " HUMAN CHECKPOINT - review the proposed research direction above.\n"
    "   • Press Enter or type APPROVE to start the literature search\n"
    "   • Type feedback (e.g. 'focus on medical imaging') to revise it\n"
    "   • Type EXIT to stop\n" + "=" * 70
)


async def console_input(prompt: str, cancellation_token: CancellationToken | None = None) -> str:
    print(CHECKPOINT_BANNER)
    return await asyncio.to_thread(input, "Your decision > ")


async def auto_approve(prompt: str, cancellation_token: CancellationToken | None = None) -> str:
    print("\n[auto-approve] Research direction approved automatically.\n")
    return "APPROVE"


async def run_research(
    topic: str,
    settings: Settings | None = None,
    *,
    mock: bool = False,
    approve_automatically: bool = False,
    input_func: InputFunc | None = None,
    fetcher: Fetcher | None = None,
    stream: bool = True,
    save: bool = True,
) -> tuple[ResearchOutput, dict]:
    """Run the full pipeline for ``topic``. Returns the output and saved file paths."""
    settings = settings or Settings()

    if mock:
        from .mock import demo_fetcher, mock_client_factory

        client_factory = mock_client_factory
        selector_client = mock_client_factory("selector")
        fetcher = fetcher or demo_fetcher
        clients_to_close: list = []
    else:
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        shared = OpenAIChatCompletionClient(model=settings.model, api_key=settings.require_api_key())

        def client_factory(_role: str) -> OpenAIChatCompletionClient:
            return shared

        selector_client = shared
        clients_to_close = [shared]

    store = PaperStore()
    search_tool = ArxivSearchTool(
        store,
        max_per_query=settings.max_papers_per_query,
        max_total=settings.max_total_papers,
        fetcher=fetcher,
    ).as_function()

    if input_func is None:
        input_func = auto_approve if approve_automatically else console_input

    agents = build_agents(client_factory, search_tool, input_func=input_func)
    team = build_team(
        agents,
        selector_client,
        max_messages=settings.max_messages,
        timeout_seconds=settings.timeout_seconds,
        max_revisions=settings.max_revisions,
    )

    task = f"Research topic: {topic}"
    try:
        if stream:
            result: TaskResult = await Console(team.run_stream(task=task))
        else:
            result = await team.run(task=task)
    finally:
        for client in clients_to_close:
            await client.close()

    output = build_output(topic, result.messages, store.all(), result.stop_reason)
    paths = save_output(output, settings.output_dir) if save else {}
    return output, paths
