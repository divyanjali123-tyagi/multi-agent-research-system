"""End-to-end run of the whole team with scripted model clients (offline)."""

import json

import pytest

from research_agents.agents import PAPER_FINDER, TOPIC_REFINER, USER
from research_agents.config import Settings
from research_agents.runner import run_research


def scripted_input(replies):
    replies = iter(replies)

    async def _input(prompt, cancellation_token=None):
        return next(replies)

    return _input


@pytest.mark.asyncio
async def test_full_pipeline_with_revision(tmp_path):
    settings = Settings(output_dir=tmp_path, openai_api_key=None)
    output, paths = await run_research(
        "retrieval augmented generation",
        settings,
        mock=True,
        input_func=scripted_input(["please emphasise evaluation", "APPROVE"]),
        stream=False,
    )

    sources = [m["source"] for m in output.transcript]
    # Human-in-the-loop: refiner -> user (feedback) -> refiner -> user (approve) -> finder
    assert sources[:6] == ["user", TOPIC_REFINER, USER, TOPIC_REFINER, USER, PAPER_FINDER]
    assert sources[-3:] == ["insight_extractor", "report_compiler", "gap_analyst"]
    assert "TERMINATE" in output.stop_reason
    assert "Revised per your feedback" in output.research_direction
    assert len(output.papers) == 3

    report = paths["report"].read_text()
    assert "Executive Summary" in report and "Research Gaps" in report
    assert not output.gap_analysis.rstrip().endswith("TERMINATE")
    assert len(json.loads(paths["papers"].read_text())) == 3


@pytest.mark.asyncio
async def test_user_can_exit_at_checkpoint(tmp_path):
    settings = Settings(output_dir=tmp_path)
    output, _ = await run_research(
        "anything", settings, mock=True, input_func=scripted_input(["exit"]), stream=False
    )
    assert [m["source"] for m in output.transcript] == ["user", TOPIC_REFINER, USER]
    assert output.papers == []


@pytest.mark.asyncio
async def test_auto_approve(tmp_path):
    output, _ = await run_research(
        "rag", Settings(output_dir=tmp_path), mock=True, approve_automatically=True, stream=False, save=False
    )
    assert output.report.startswith("# ")


def test_missing_api_key_message():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Settings(openai_api_key=None).require_api_key()
