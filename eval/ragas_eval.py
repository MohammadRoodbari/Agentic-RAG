"""Online, per-query Ragas scoring.

Scores a single live Q&A turn (Faithfulness, AnswerRelevancy) right after
the RAG pipeline answers it, and attaches the scores to the matching
Phoenix trace span. See score_predictions.py for the offline, reference-
based batch scoring of the held-out QA set.
"""

from __future__ import annotations

import asyncio

from phoenix.client import Client
from ragas.dataset_schema import SingleTurnSample

from eval.ragas_factory import build_client, build_metrics

_client = build_client()
_metrics = build_metrics(_client)
_faithfulness = _metrics["faithfulness"]
_answer_relevancy = _metrics["answer_relevancy"]

_phoenix_client = Client()


async def _score_all(sample: SingleTurnSample) -> dict[str, float]:
    faithfulness_result = await _faithfulness.ascore(
        user_input=sample.user_input,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts,
    )
    relevancy_result = await _answer_relevancy.ascore(
        user_input=sample.user_input,
        response=sample.response,
    )

    return {
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": relevancy_result.value,
    }


def evaluate_and_log(
    *,
    span_id: str,
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """Score one Q&A turn with Ragas and attach the scores to the Phoenix span."""
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    scores = asyncio.run(_score_all(sample))
    for name, value in scores.items():
        _phoenix_client.spans.add_span_annotation(
            span_id=span_id,
            annotation_name=name,
            annotator_kind="LLM",
            score=float(value),
        )
    return scores
