"""arXiv retrieval tool exposed to the PaperFinder agent.

The tool searches arXiv for one or more queries, de-duplicates the results,
stores them in a shared :class:`PaperStore` (so the final report can cite
them and export JSON) and returns a compact, LLM-friendly text listing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field

import arxiv

logger = logging.getLogger(__name__)

ABSTRACT_CHAR_LIMIT = 900


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    summary: str
    url: str
    pdf_url: str | None = None
    categories: list[str] = field(default_factory=list)
    query: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def short_authors(self, limit: int = 3) -> str:
        if len(self.authors) <= limit:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:limit]) + " et al."


@dataclass
class PaperStore:
    """Shared, de-duplicated collection of papers found during a run."""

    papers: dict[str, Paper] = field(default_factory=dict)

    def add(self, paper: Paper) -> bool:
        if paper.arxiv_id in self.papers:
            return False
        self.papers[paper.arxiv_id] = paper
        return True

    def all(self) -> list[Paper]:
        return list(self.papers.values())

    def __len__(self) -> int:
        return len(self.papers)


# A fetcher takes (query, max_results) and returns Papers. Swappable for tests.
Fetcher = Callable[[str, int], list[Paper]]


def _result_to_paper(result: arxiv.Result, query: str) -> Paper:
    return Paper(
        arxiv_id=result.get_short_id(),
        title=" ".join(result.title.split()),
        authors=[a.name for a in result.authors],
        published=result.published.date().isoformat() if result.published else "",
        summary=" ".join(result.summary.split()),
        url=result.entry_id,
        pdf_url=result.pdf_url,
        categories=list(result.categories or []),
        query=query,
    )


def arxiv_fetcher(query: str, max_results: int) -> list[Paper]:
    """Default fetcher backed by the public arXiv API (blocking)."""
    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [_result_to_paper(r, query) for r in client.results(search)]


def format_papers(papers: Sequence[Paper]) -> str:
    """Render papers as a numbered list the downstream agents can read."""
    if not papers:
        return "No papers found."
    lines: list[str] = []
    for i, p in enumerate(papers, start=1):
        abstract = p.summary
        if len(abstract) > ABSTRACT_CHAR_LIMIT:
            abstract = abstract[:ABSTRACT_CHAR_LIMIT].rsplit(" ", 1)[0] + " ..."
        lines.append(
            f"[{i}] {p.title}\n"
            f"    arXiv: {p.arxiv_id} | Published: {p.published} | Authors: {p.short_authors()}\n"
            f"    URL: {p.url}\n"
            f"    Abstract: {abstract}"
        )
    return "\n\n".join(lines)


class ArxivSearchTool:
    """Callable wrapper turned into an AutoGen tool via ``as_function``."""

    def __init__(
        self,
        store: PaperStore,
        max_per_query: int = 5,
        max_total: int = 12,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.store = store
        self.max_per_query = max_per_query
        self.max_total = max_total
        self.fetcher = fetcher or arxiv_fetcher

    async def search(self, queries: Iterable[str]) -> str:
        found: list[Paper] = []
        errors: list[str] = []
        for query in [q.strip() for q in queries if q and q.strip()]:
            if len(self.store) >= self.max_total:
                break
            try:
                results = await asyncio.to_thread(self.fetcher, query, self.max_per_query)
            except Exception as exc:  # network errors, rate limits, ...
                logger.warning("arXiv query %r failed: %s", query, exc)
                errors.append(f"Query {query!r} failed: {exc}")
                continue
            for paper in results:
                if len(self.store) >= self.max_total:
                    break
                if self.store.add(paper):
                    found.append(paper)

        header = f"Retrieved {len(found)} unique papers from arXiv."
        body = format_papers(found)
        if errors:
            body += "\n\nErrors:\n" + "\n".join(errors)
        return f"{header}\n\n{body}"

    def as_function(self):
        """Return an async function with a clean signature/docstring for the LLM."""

        async def search_arxiv(queries: list[str]) -> str:
            """Search arXiv for academic papers.

            Args:
                queries: 2-4 focused arXiv search queries. Field prefixes such as
                    ti: (title), abs: (abstract) and cat: (category, e.g. cat:cs.CL)
                    and boolean operators (AND, OR, ANDNOT) are supported.

            Returns:
                A numbered list of unique papers with title, id, date, authors,
                URL and abstract.
            """
            return await self.search(queries)

        return search_arxiv
