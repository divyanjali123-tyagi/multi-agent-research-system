"""Tests for the arXiv tool using a fake fetcher (no network)."""

import pytest

from research_agents.tools.arxiv_tool import ArxivSearchTool, Paper, PaperStore, format_papers


def make_paper(pid: str, title: str = "Title") -> Paper:
    return Paper(
        arxiv_id=pid,
        title=title,
        authors=["A", "B", "C", "D"],
        published="2025-01-01",
        summary="word " * 400,
        url=f"https://arxiv.org/abs/{pid}",
    )


def fake_fetcher(query, max_results):
    catalogue = {
        "q1": [make_paper("1"), make_paper("2")],
        "q2": [make_paper("2"), make_paper("3")],  # "2" is a duplicate
    }
    if query == "boom":
        raise ConnectionError("arXiv unavailable")
    return catalogue.get(query, [])[:max_results]


@pytest.mark.asyncio
async def test_search_deduplicates_across_queries():
    store = PaperStore()
    tool = ArxivSearchTool(store, max_per_query=5, max_total=10, fetcher=fake_fetcher)
    out = await tool.search(["q1", "q2"])
    assert len(store) == 3
    assert "Retrieved 3 unique papers" in out


@pytest.mark.asyncio
async def test_search_respects_total_cap():
    store = PaperStore()
    tool = ArxivSearchTool(store, max_per_query=5, max_total=2, fetcher=fake_fetcher)
    await tool.search(["q1", "q2"])
    assert len(store) == 2


@pytest.mark.asyncio
async def test_search_reports_errors_but_continues():
    store = PaperStore()
    tool = ArxivSearchTool(store, fetcher=fake_fetcher)
    out = await tool.search(["boom", "q1"])
    assert len(store) == 2
    assert "failed" in out


@pytest.mark.asyncio
async def test_tool_function_signature_for_llm():
    fn = ArxivSearchTool(PaperStore(), fetcher=fake_fetcher).as_function()
    assert fn.__name__ == "search_arxiv"
    assert "arXiv" in fn.__doc__
    assert "Retrieved" in await fn(["q1"])


def test_format_truncates_abstracts_and_authors():
    text = format_papers([make_paper("9", "Long one")])
    assert "et al." in text
    assert text.endswith("...")
    assert format_papers([]) == "No papers found."
