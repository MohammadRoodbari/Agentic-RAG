"""Shared Ragas configuration.

Centralizes the OpenAI client, judge LLM, embeddings, and metric objects
used across every evaluation entry point in this project:

- eval/ragas_eval.py        -- online, per-query scoring (Faithfulness,
                                AnswerRelevancy), logged to Phoenix.
- eval/score_predictions.py -- offline, batch scoring against a
                                reference-labeled QA set (all four metrics).

Building both from one factory keeps their configuration (judge model,
embedding model, strictness) identical, so scores from the two paths
stay comparable instead of silently drifting apart.
"""

from __future__ import annotations

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.config import settings

ANSWER_RELEVANCY_STRICTNESS = 2


def build_client() -> AsyncOpenAI:
    """Create the OpenAI client shared by the judge LLM and embeddings."""
    return AsyncOpenAI(api_key=settings.require_api_key(), base_url=settings.base_url)


def build_metrics(client: AsyncOpenAI | None = None) -> dict:
    """Build every Ragas metric used in this project, keyed by name.

    Args:
        client: Optional pre-built OpenAI client to reuse (e.g. so a
            caller can hold onto it and close it later). If omitted, a
            new client is created via `build_client()`.

    Returns:
        Dict with keys "faithfulness", "answer_relevancy",
        "context_precision", "context_recall".
    """
    client = client or build_client()
    judge = llm_factory(settings.openai_model, client=client)
    embeddings = OpenAIEmbeddings(client=client, model=settings.embedding_model)

    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge, embeddings=embeddings, strictness=ANSWER_RELEVANCY_STRICTNESS
        ),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }
