"""Step 2 of the eval: score eval/predictions.json with Ragas.

    # from the project root, after running generate_predictions.py:
    python -m eval.score_predictions

Scores every answerable question on faithfulness, answer relevancy, context
precision, and context recall, prints per-question and aggregate results,
and writes eval/ragas_results.csv. Out-of-scope questions are graded
separately with a simple refusal check (no LLM judge needed).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from app.config import settings
from eval.ragas_factory import build_metrics

_HERE = Path(__file__).parent
PREDICTIONS_PATH = _HERE / "predictions.json"
OUT_CSV = _HERE / "ragas_results.csv"


async def _score_one(metrics: dict, sample: dict) -> dict:
    """Run all metrics for a single Q&A sample. Returns metric_name -> float."""
    faithfulness_result, relevancy_result, precision_result, recall_result = await asyncio.gather(
        metrics["faithfulness"].ascore(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"],
        ),
        metrics["answer_relevancy"].ascore(
            user_input=sample["user_input"],
            response=sample["response"],
        ),
        metrics["context_precision"].ascore(
            user_input=sample["user_input"],
            retrieved_contexts=sample["retrieved_contexts"],
            reference=sample["reference"],
        ),
        metrics["context_recall"].ascore(
            user_input=sample["user_input"],
            retrieved_contexts=sample["retrieved_contexts"],
            reference=sample["reference"],
        ),
    )
    return {
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": relevancy_result.value,
        "context_precision": precision_result.value,
        "context_recall": recall_result.value,
    }


async def _score_all(metrics: dict, samples: list[dict]) -> list[dict]:
    # Sequential to keep judge-API rate limits sane; switch to
    # asyncio.gather(*[_score_one(...) for ...]) if you want concurrency.
    return [await _score_one(metrics, s) for s in samples]


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise SystemExit(
            f"{PREDICTIONS_PATH} not found. Run `python -m eval.generate_predictions` first."
        )

    preds = json.loads(PREDICTIONS_PATH.read_text("utf-8"))
    answerable = [p for p in preds if not p.get("out_of_scope")]
    oos = [p for p in preds if p.get("out_of_scope")]

    samples = [
        {
            "user_input": p["question"],
            "response": p["answer"],
            "retrieved_contexts": p["contexts"],
            "reference": p["reference"],
        }
        for p in answerable
    ]

    metrics = build_metrics()

    print(f"Scoring {len(samples)} answerable questions with ragas (judge={settings.openai_model})...\n")

    scored = asyncio.run(_score_all(metrics, samples))
    df = pd.DataFrame(scored)
    metric_cols = list(df.columns)

    # Per-question scores, keyed back to the QA ids.
    print("Per-question scores")
    print("-" * 72)
    for p, row in zip(answerable, scored):
        scores = "  ".join(f"{c}={row[c]:.2f}" for c in metric_cols)
        print(f"{p['id']:<26} {scores}")

    print("\nAggregate (mean across questions)")
    print("-" * 72)
    for c in metric_cols:
        print(f"{c:<22} {df[c].mean():.3f}")

    # Safety / grounding result for the out-of-scope question(s).
    if oos:
        print("\nOut-of-scope refusal check (no LLM judge)")
        print("-" * 72)
        for p in oos:
            verdict = "PASS (refused)" if p["refused"] else "FAIL (answered anyway!)"
            print(f"{p['id']:<26} {verdict}")

    df.insert(0, "id", [p["id"] for p in answerable])
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote per-question results -> {OUT_CSV}")


if __name__ == "__main__":
    main()
