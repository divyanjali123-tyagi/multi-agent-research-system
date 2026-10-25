"""Command-line interface: ``research-agents "your topic"`` or ``python -m research_agents``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .config import Settings
from .runner import run_research


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-agents",
        description="Multi-agent academic research automation with AutoGen, OpenAI and arXiv.",
    )
    parser.add_argument("topic", nargs="*", help="Research topic (prompted for if omitted).")
    parser.add_argument("--model", help="OpenAI model name (default: $OPENAI_MODEL or gpt-4o-mini).")
    parser.add_argument("--max-papers", type=int, help="Maximum total papers to retrieve.")
    parser.add_argument("--per-query", type=int, help="Maximum papers per arXiv query.")
    parser.add_argument("--max-messages", type=int, help="Hard cap on messages before stopping.")
    parser.add_argument("--output-dir", type=Path, help="Where to save reports (default: reports/).")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip the human approval checkpoint (non-interactive runs, CI).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Offline demo: scripted agents + synthetic papers, no API key needed.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    topic = " ".join(args.topic).strip()
    if not topic:
        topic = "retrieval-augmented generation" if args.mock else input("Research topic > ").strip()
    if not topic:
        print("A research topic is required.", file=sys.stderr)
        return 2

    settings = Settings()
    if args.model:
        settings.model = args.model
    if args.max_papers:
        settings.max_total_papers = args.max_papers
    if args.per_query:
        settings.max_papers_per_query = args.per_query
    if args.max_messages:
        settings.max_messages = args.max_messages
    if args.output_dir:
        settings.output_dir = args.output_dir

    try:
        output, paths = asyncio.run(
            run_research(
                topic,
                settings,
                mock=args.mock,
                approve_automatically=args.auto_approve,
            )
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print("\n" + "=" * 70)
    print(f"Run finished. Stop reason: {output.stop_reason}")
    print(f"Papers retrieved: {len(output.papers)}")
    for name, path in paths.items():
        print(f"  {name:<10} -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
