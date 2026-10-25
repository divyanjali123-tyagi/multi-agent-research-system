"""Assemble and persist the final research report from the team transcript."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage

from . import prompts
from .agents import GAP_ANALYST, REPORT_COMPILER, TOPIC_REFINER, USER
from .tools.arxiv_tool import Paper


@dataclass
class ResearchOutput:
    topic: str
    research_direction: str = ""
    report: str = ""
    gap_analysis: str = ""
    papers: list[Paper] = field(default_factory=list)
    stop_reason: str | None = None
    transcript: list[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        parts = [
            self.report.strip() or f"# Research Report: {self.topic}\n\n_No report was produced._",
            self.gap_analysis.strip(),
            "---",
            "## Appendix: Approved Research Direction",
            self.research_direction.strip() or "_n/a_",
            "",
            f"_Generated {datetime.now():%Y-%m-%d %H:%M} by the Multi-Agent Research "
            f"Automation System. Stop reason: {self.stop_reason}._",
        ]
        return "\n\n".join(p for p in parts if p is not None)


def _strip_terminate(text: str) -> str:
    return re.sub(rf"\s*\b{prompts.TERMINATE_KEYWORD}\b\s*$", "", text).rstrip()


def _content(m: BaseChatMessage) -> str:
    return m.to_text()


def build_output(
    topic: str,
    messages: Sequence[BaseAgentEvent | BaseChatMessage],
    papers: list[Paper],
    stop_reason: str | None,
) -> ResearchOutput:
    chat = [m for m in messages if isinstance(m, BaseChatMessage)]
    out = ResearchOutput(topic=topic, papers=papers, stop_reason=stop_reason)

    # The approved direction is the last refiner proposal before the final user reply.
    for m in chat:
        if m.source == TOPIC_REFINER:
            out.research_direction = _content(m)
        elif m.source == REPORT_COMPILER:
            out.report = _content(m)
        elif m.source == GAP_ANALYST:
            out.gap_analysis = _strip_terminate(_content(m))

    out.transcript = [{"source": m.source, "type": m.type, "content": _content(m)} for m in chat]
    return out


def slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "research"


def save_output(output: ResearchOutput, directory: Path) -> dict[str, Path]:
    """Write report.md, papers.json and transcript.json into a timestamped folder."""
    run_dir = directory / f"{datetime.now():%Y%m%d-%H%M%S}-{slugify(output.topic)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "report": run_dir / "report.md",
        "papers": run_dir / "papers.json",
        "transcript": run_dir / "transcript.json",
    }
    paths["report"].write_text(output.to_markdown(), encoding="utf-8")
    paths["papers"].write_text(json.dumps([p.to_dict() for p in output.papers], indent=2), encoding="utf-8")
    paths["transcript"].write_text(
        json.dumps(
            {"topic": output.topic, "stop_reason": output.stop_reason, "messages": output.transcript},
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


__all__ = ["USER", "ResearchOutput", "build_output", "save_output", "slugify"]
