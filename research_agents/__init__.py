"""Multi-Agent Research Automation System built on AutoGen, OpenAI and arXiv."""

from .config import Settings
from .runner import run_research

__all__ = ["Settings", "run_research"]
__version__ = "1.0.0"
