"""Offline demo mode: scripted model clients and a fake arXiv fetcher.

``--mock`` lets anyone clone the repo and watch the full multi-agent workflow
(including the approval checkpoint) without an OpenAI key or network access.
All papers here are clearly-labelled synthetic demo data.
"""

from __future__ import annotations

import json

from autogen_core import FunctionCall
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

from .agents import (
    GAP_ANALYST,
    INSIGHT_EXTRACTOR,
    PAPER_FINDER,
    REPORT_COMPILER,
    TOPIC_REFINER,
)
from .tools.arxiv_tool import Paper

MOCK_MODEL_INFO = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=False,
    family="unknown",
    structured_output=False,
)

DEMO_PAPERS: list[Paper] = [
    Paper(
        arxiv_id="demo.0001",
        title="[DEMO] Retrieval-Augmented Generation for Knowledge-Intensive QA",
        authors=["A. Researcher", "B. Scientist"],
        published="2025-01-15",
        summary=(
            "We study retrieval-augmented generation (RAG) for open-domain question answering. "
            "A dense retriever supplies passages to a sequence-to-sequence generator. RAG improves "
            "exact match over closed-book baselines by 9 points, but performance drops sharply when "
            "retrieved passages are noisy."
        ),
        url="https://arxiv.org/abs/demo.0001",
        categories=["cs.CL"],
    ),
    Paper(
        arxiv_id="demo.0002",
        title="[DEMO] Evaluating Faithfulness of Retrieval-Augmented LLMs",
        authors=["C. Analyst", "D. Engineer", "E. Student", "F. Advisor"],
        published="2025-03-02",
        summary=(
            "We introduce a benchmark measuring whether RAG outputs are supported by retrieved "
            "evidence. Across 6 models, 18-31% of answers contain unsupported claims. Faithfulness "
            "correlates weakly with answer accuracy, suggesting accuracy-only evaluation is insufficient."
        ),
        url="https://arxiv.org/abs/demo.0002",
        categories=["cs.CL", "cs.AI"],
    ),
    Paper(
        arxiv_id="demo.0003",
        title="[DEMO] Adaptive Retrieval: Deciding When to Retrieve",
        authors=["G. Builder"],
        published="2025-06-20",
        summary=(
            "Always retrieving wastes compute and can inject distractors. We train a lightweight "
            "classifier that predicts when retrieval helps, cutting retrieval calls by 40% while "
            "matching accuracy on multi-hop QA. Evaluation is limited to English Wikipedia."
        ),
        url="https://arxiv.org/abs/demo.0003",
        categories=["cs.CL", "cs.IR"],
    ),
]


def demo_fetcher(query: str, max_results: int) -> list[Paper]:
    return [Paper(**{**p.to_dict(), "query": query}) for p in DEMO_PAPERS[:max_results]]


def _tool_call(queries: list[str]) -> CreateResult:
    return CreateResult(
        finish_reason="function_calls",
        content=[
            FunctionCall(
                id="call_demo_1",
                name="search_arxiv",
                arguments=json.dumps({"queries": queries}),
            )
        ],
        usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
        cached=False,
    )


_REFINER_V1 = """## Refined Research Direction
How do retrieval-augmented generation (RAG) systems trade off answer accuracy against faithfulness to retrieved evidence?

## Scope
- In scope: RAG for question answering, faithfulness evaluation, adaptive retrieval
- Out of scope: pure fine-tuning approaches, multimodal retrieval

## Proposed arXiv Queries
1. `abs:"retrieval augmented generation" AND cat:cs.CL`
2. `abs:"faithfulness" AND abs:"retrieval"`
3. `ti:"adaptive retrieval"`

## Key Questions
1. How much does retrieval improve QA accuracy?
2. How often do RAG systems produce unsupported claims?
3. Can systems decide when retrieval is necessary?

Reply APPROVE to proceed, or tell me what to change."""

_REFINER_V2 = _REFINER_V1.replace(
    "## Key Questions",
    "_Revised per your feedback: added emphasis on evaluation methodology._\n\n## Key Questions",
)

_INSIGHTS = """### [1] [DEMO] Retrieval-Augmented Generation for Knowledge-Intensive QA (demo.0001)
- **Problem:** Closed-book LLMs lack up-to-date factual knowledge.
- **Method:** Dense retriever + seq2seq generator.
- **Key findings:** +9 EM over closed-book baselines.
- **Limitations:** Sensitive to noisy retrieved passages.

### [2] [DEMO] Evaluating Faithfulness of Retrieval-Augmented LLMs (demo.0002)
- **Problem:** Accuracy metrics ignore whether answers are grounded.
- **Method:** New faithfulness benchmark across 6 models.
- **Key findings:** 18-31% of answers contain unsupported claims.
- **Limitations:** Benchmark scope not stated in abstract.

### [3] [DEMO] Adaptive Retrieval: Deciding When to Retrieve (demo.0003)
- **Problem:** Always-on retrieval wastes compute and adds distractors.
- **Method:** Classifier predicting retrieval utility.
- **Key findings:** 40% fewer retrieval calls at equal accuracy.
- **Limitations:** English Wikipedia only.

## Cross-Paper Themes
- Retrieval quality strongly drives output quality [1][3]
- Accuracy and faithfulness are only weakly correlated [2]
- Efficiency via selective retrieval [3]"""

_REPORT = """# Accuracy vs. Faithfulness in Retrieval-Augmented Generation (Demo Report)

## Executive Summary
Retrieval-augmented generation improves factual QA accuracy over closed-book models [1]. However, a substantial share of RAG answers are not supported by the retrieved evidence [2], and accuracy alone is a poor proxy for faithfulness. Adaptive retrieval reduces cost without hurting accuracy [3]. Together these results suggest evaluation must jointly measure accuracy, grounding and efficiency.

## Background
RAG couples a retriever with a generator so models can consult external knowledge at inference time.

## Thematic Review
**Retrieval quality as the bottleneck.** Gains depend on clean passages; noise degrades answers [1], and indiscriminate retrieval injects distractors [3].

**Faithfulness is a separate axis.** Unsupported claims persist even in accurate systems [2].

## Comparison Table
| Paper | Approach | Data/Setting | Key Result |
|---|---|---|---|
| [1] | Dense retriever + seq2seq | Open-domain QA | +9 EM |
| [2] | Faithfulness benchmark | 6 LLMs | 18-31% unsupported |
| [3] | Retrieval-utility classifier | Multi-hop QA, Wikipedia | -40% retrieval calls |

## Answers to Key Questions
1. Retrieval improves accuracy substantially [1].
2. Unsupported claims are common (18-31%) [2].
3. Yes, learned classifiers can gate retrieval [3].

## References
[1] [DEMO] Retrieval-Augmented Generation for Knowledge-Intensive QA - demo.0001 - https://arxiv.org/abs/demo.0001
[2] [DEMO] Evaluating Faithfulness of Retrieval-Augmented LLMs - demo.0002 - https://arxiv.org/abs/demo.0002
[3] [DEMO] Adaptive Retrieval: Deciding When to Retrieve - demo.0003 - https://arxiv.org/abs/demo.0003"""

_GAPS = """## Research Gaps
1. **Joint accuracy-faithfulness objectives.** Evidence: weak correlation [2]. Matters because optimizing accuracy may not improve grounding.
2. **Robustness to noisy retrieval.** Evidence: [1][3]. Matters for real-world corpora.
3. **Multilingual adaptive retrieval.** Evidence: English-only evaluation [3].

## Suggested Future Directions
1. Train generators with a faithfulness reward and report both metrics.
2. Benchmark RAG under controlled retrieval-noise levels.
3. Extend retrieval-gating classifiers to multilingual corpora.

## Confidence & Caveats
Based on three synthetic demo abstracts only.

TERMINATE"""


def mock_client_factory(role: str) -> ChatCompletionClient:
    """Return a scripted client per role (used with ``--mock``)."""
    scripts: dict[str, list] = {
        TOPIC_REFINER: [_REFINER_V1, _REFINER_V2, _REFINER_V2, _REFINER_V2],
        PAPER_FINDER: [
            _tool_call(
                [
                    'abs:"retrieval augmented generation" AND cat:cs.CL',
                    'abs:"faithfulness" AND abs:"retrieval"',
                ]
            ),
            _tool_call(['ti:"adaptive retrieval"']),
        ],
        INSIGHT_EXTRACTOR: [_INSIGHTS],
        REPORT_COMPILER: [_REPORT],
        GAP_ANALYST: [_GAPS, _GAPS],
    }
    return ReplayChatCompletionClient(scripts.get(role, ["TERMINATE"]), model_info=MOCK_MODEL_INFO)
