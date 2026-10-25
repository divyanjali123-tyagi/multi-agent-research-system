"""Runtime configuration loaded from environment variables / a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


@dataclass
class Settings:
    """All tunable knobs for a research run."""

    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    max_papers_per_query: int = field(default_factory=lambda: _int_env("MAX_PAPERS_PER_QUERY", 5))
    max_total_papers: int = field(default_factory=lambda: _int_env("MAX_TOTAL_PAPERS", 12))
    max_messages: int = field(default_factory=lambda: _int_env("MAX_MESSAGES", 40))
    timeout_seconds: int = field(default_factory=lambda: _int_env("RUN_TIMEOUT_SECONDS", 900))
    max_revisions: int = field(default_factory=lambda: _int_env("MAX_TOPIC_REVISIONS", 3))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("REPORTS_DIR", "reports")))

    def require_api_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or run with --mock for an offline demo."
            )
        return self.openai_api_key
