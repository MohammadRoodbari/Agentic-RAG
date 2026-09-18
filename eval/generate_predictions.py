"""Step 1 of the eval: run the REAL RAG pipeline over the held-out QA set and
dump predictions to eval/predictions.json.

    # from the project root, with the serving venv active:
    python -m eval.generate_predictions

Each prediction captures everything ragas needs (id, question, generated
answer, retrieved contexts, reference) plus a refusal flag for out-of-scope
questions. These runs are tagged "eval" in LangSmith so they don't mix with
live traffic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agent.graph import qa_graph

_HERE = Path(__file__).parent
QA_SET_PATH = _HERE / "qa_set.json"
OUT_PATH = _HERE / "predictions.json"

# Phrases that signal the model correctly declined to answer (used only to grade
# the out-of-scope question -- a confident answer there would be a hallucination).
REFUSAL_MARKERS = [
    "cannot", "can't", "not enough", "no information", "insufficient",
    "couldn't find", "could not find", "don't have", "do not have",
    "not available", "no relevant", "not contain", "doesn't contain",
    "do not contain", "unable", "not provided", "outside",
]


def _looks_like_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def _require(entry: dict[str, Any], key: str) -> Any:
    """Fetch a required field from a qa_set.json entry, with a clear error
    that points at the offending entry instead of a bare KeyError."""
    try:
        return entry[key]
    except KeyError as exc:
        raise KeyError(f"qa_set.json entry is missing required field '{key}': {entry}") from exc


def main() -> None:
    qa_set = json.loads(QA_SET_PATH.read_text("utf-8"))

    predictions = []
    for q in qa_set:
        qid = _require(q, "id")
        question = _require(q, "question")
        reference = _require(q, "reference")

        result = qa_graph.invoke(
            {
                "question": question,
                "retrieved_chunks": [],
                "answer": "",
                "citations": [],
                "error": None,
                # "api_key": None,  # use the server's OPENAI_API_KEY
            }
        )
        if result.get("error"):
            raise RuntimeError(f"Pipeline error on '{qid}': {result['error']}")

        answer = result["answer"]
        contexts = [c["text"] for c in result["retrieved_chunks"]]

        predictions.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "reference": reference,
                "out_of_scope": q.get("out_of_scope", False),
                "refused": _looks_like_refusal(answer),
            }
        )
        tag = "OOS" if q.get("out_of_scope") else "   "
        print(f"[{tag}] {qid:<26} {len(contexts)} chunks")

    OUT_PATH.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nWrote {len(predictions)} predictions -> {OUT_PATH}")

    # Quick safety check on the out-of-scope question(s) (no LLM judge needed).
    for p in predictions:
        if p["out_of_scope"]:
            verdict = "PASS (refused)" if p["refused"] else "FAIL (answered anyway!)"
            print(f"Out-of-scope grounding check [{p['id']}]: {verdict}")


if __name__ == "__main__":
    main()
