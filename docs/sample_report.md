# Accuracy vs. Faithfulness in Retrieval-Augmented Generation (Demo Report)

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
[3] [DEMO] Adaptive Retrieval: Deciding When to Retrieve - demo.0003 - https://arxiv.org/abs/demo.0003

## Research Gaps
1. **Joint accuracy-faithfulness objectives.** Evidence: weak correlation [2]. Matters because optimizing accuracy may not improve grounding.
2. **Robustness to noisy retrieval.** Evidence: [1][3]. Matters for real-world corpora.
3. **Multilingual adaptive retrieval.** Evidence: English-only evaluation [3].

## Suggested Future Directions
1. Train generators with a faithfulness reward and report both metrics.
2. Benchmark RAG under controlled retrieval-noise levels.
3. Extend retrieval-gating classifiers to multilingual corpora.

## Confidence & Caveats
Based on three synthetic demo abstracts only.

---

## Appendix: Approved Research Direction

## Refined Research Direction
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

Reply APPROVE to proceed, or tell me what to change.



_Generated 2026-10-25 03:40 by the Multi-Agent Research Automation System. Stop reason: Text 'TERMINATE' mentioned._