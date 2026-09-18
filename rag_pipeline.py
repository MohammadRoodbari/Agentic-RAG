"""
RAG query pipeline for the Compliance RAG Agent.

Runs a question through the QA graph, prints the answer and citations,
and scores the response with Ragas metrics logged to Phoenix.

Usage:
    python rag_pipeline.py "your question here"
"""

from __future__ import annotations

import sys
from typing import Any

from app.agent.graph import qa_graph
from app.services.observability import setup_tracing
from eval.ragas_eval import evaluate_and_log

_tracer_provider = setup_tracing()
_tracer = _tracer_provider.get_tracer(__name__)


def build_initial_state(question: str) -> dict[str, Any]:
    """Build the initial graph state for a new question."""
    return {
        "question": question,
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "error": None,
    }


def run_qa(question: str) -> tuple[dict[str, Any], str]:
    """Invoke the QA graph inside a traced span.

    Returns:
        A tuple of (result_state, span_id).

    Raises:
        RuntimeError: If the graph reports an error.
    """
    with _tracer.start_as_current_span("RAG") as span:
        result = qa_graph.invoke(build_initial_state(question))
        span_id = format(span.get_span_context().span_id, "016x")

    if result.get("error"):
        raise RuntimeError(result["error"])

    return result, span_id


def extract_contexts(retrieved_chunks: list[dict[str, Any]]) -> list[str]:
    """Pull the raw text out of retrieved chunks for Ragas scoring.

    Adjust the key below if retrieved_chunks are shaped differently
    (e.g. c.page_content, c["content"], ...).
    """
    return [c["text"] for c in retrieved_chunks]


def print_results(result: dict[str, Any], scores: dict[str, float]) -> None:
    """Pretty-print the answer, citations, and Ragas scores."""
    print("\n========================")
    print("Answer")
    print("========================")
    print(result["answer"])

    print("\n========================")
    print("Citations")
    print("========================")
    for i, citation in enumerate(result["citations"], 1):
        print(f"[{i}] {citation['source']} (page {citation['page']})")

    print("\n========================")
    print("Ragas scores")
    print("========================")
    for name, value in scores.items():
        print(f"{name}: {value:.3f}")


def ask(question: str) -> dict[str, float]:
    """Run the QA graph on a question and score the answer with Ragas."""
    result, span_id = run_qa(question)
    contexts = extract_contexts(result.get("retrieved_chunks", []))

    scores = evaluate_and_log(
        span_id=span_id,
        question=question,
        answer=result["answer"],
        contexts=contexts,
    )

    print_results(result, scores)
    return scores


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print('Usage: python rag_pipeline.py "your question"')
        return 1

    try:
        ask(argv[0])
    except RuntimeError as exc:
        print(f"Error: QA pipeline failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())