# Agentic-RAG

A **Retrieval-Augmented Generation** pipeline that answers natural-language questions about compliance and regulatory documents — with **source citations**. It runs as two command-line pipelines (ingest + query), built on **LangGraph**, **Weaviate** hybrid search, **parent-child chunking**, and a **Redis**-backed context store, with **Ragas** evaluation and **Phoenix** tracing throughout.

---

## Features

- **Parent-child chunking** — documents are split into large parent windows for context, then each parent is split further into small child windows. Only the children are embedded and searched, but the LLM is given the full parent text, so retrieval stays precise while generation stays well-grounded.
- **Hybrid retrieval via Weaviate** — native hybrid search (BM25 + vector) over child chunks, with `hybrid_alpha` tuning the BM25/vector balance (`0` = pure BM25, `1` = pure vector).
- **Redis-backed parent store** — child-chunk hits are hydrated back to their full parent text via a fast Redis lookup keyed by parent id.
- **Agentic pipeline (LangGraph)** — a `StateGraph` (validate → retrieve → generate → format) that short-circuits to an error state at any stage.
- **Grounded answers with citations** — every answer references the exact source document and page (e.g. `[1] gdpr.pdf, Page 12`).
- **Observability via Phoenix** — every pipeline run is traced end-to-end; Ragas scores are attached to the matching trace as span annotations.
- **Evaluation suite** — online, per-query Ragas scoring on every live question, plus offline batch scoring against a curated, reference-labeled QA set — both built from one shared metrics factory so they can't drift out of sync.

---

## Architecture

```
                        ┌───────────────────────────────────────────────────────────┐
                        │                     Ingest Pipeline                       │
 PDF / TXT / MD / CSV ─►│  Extract text ─► Parent chunk ─► Child chunk ─► Embed     │
                        │                                                     │      │
                        │           Weaviate  ◄── children (vectors + BM25)  │      │
                        │           Redis     ◄── parents  (full text, by id)◄──────┘
                        └───────────────────────────────────────────────────────────┘

                        ┌───────────────────────────────────────────────────────────┐
                        │                 Query Pipeline (LangGraph)                │
 Question            ──►│  Validate ─► Retrieve ─► Generate ─► Format               │
                        │                 │            │           │                │
                        │                 │            │           └─► citations    │
                        │                 │            └─► LLM answer, grounded     │
                        │                 │                 in hydrated parents     │
                        │                 └─► hybrid search over children (Weaviate)│
                        │                     top `retrieval_top_k` → hydrate       │
                        │                     parents from Redis → keep             │
                        │                     `final_top_k` for generation          │
                        └───────────────────────────┬───────────────────────────────┘
                                                      │ traced end-to-end
                                                      ▼
                                                   Phoenix
                                          (+ optional per-turn Ragas
                                           scores as span annotations)
```

---

## Quick start

**Prerequisites:** Python 3.11+, a running Weaviate instance, a running Redis instance.

```bash
docker compose up -d
pip install -r requirements.txt
```

**Ingest a document:**
```bash
python ingest_pipeline.py path/to/document.pdf
```

**Ask a question:**
```bash
python rag_pipeline.py "What are the lawful bases for processing personal data under GDPR?"
```

Each run prints the answer, its citations, and live Ragas scores (faithfulness, answer relevancy) — and the full trace shows up in Phoenix at `PHOENIX_URL`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** API key for your OpenAI-compatible provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for answer generation and as the Ragas judge |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for chunks and queries |
| `BASE_URL` | — | OpenAI-compatible API base URL — point this at any compatible provider |
| `MAX_PDF_PAGES` | `100` | Hard cap on pages read from an ingested PDF |
| `RETRIEVAL_TOP_K` | `10` | Child chunks fetched from Weaviate per query |
| `FINAL_TOP_K` | `3` | Parent chunks kept (after hydration) and sent to the LLM |
| `WEAVIATE_HTTP_HOST` / `WEAVIATE_HTTP_PORT` | `localhost` / `8080` | Weaviate REST endpoint |
| `WEAVIATE_GRPC_HOST` / `WEAVIATE_GRPC_PORT` | `localhost` / `50051` | Weaviate gRPC endpoint |
| `WEAVIATE_COLLECTION` | `DocumentChunk` | Weaviate collection storing child chunks |
| `HYBRID_ALPHA` | `0.5` | Hybrid search weight: `0` = pure BM25, `1` = pure vector |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis instance storing parent chunk text, keyed by parent id |
| `REDIS_PARENT_TTL_SECONDS` | unset (no expiry) | Optional TTL for parent chunks in Redis |
| `PHOENIX_URL` | `http://127.0.0.1:6006` | Arize Phoenix endpoint for tracing + span annotations |
| `PARENT_CHUNK_SIZE` / `PARENT_CHUNK_OVERLAP` | `1700` / `200` | Size / overlap of parent chunks |
| `CHILD_CHUNK_SIZE` / `CHILD_CHUNK_OVERLAP` | `300` / `40` | Size / overlap of child chunks (what's actually embedded and searched) |

---

## Evaluation

Two complementary Ragas paths, both built from the same `eval/ragas_factory.py` so their judge model, embeddings, and metric configuration never drift apart:

**1. Online — every live question.** `rag_pipeline.py` scores each answer on `faithfulness` and `answer_relevancy` right after it's generated (no reference answer needed) and attaches the scores to that question's Phoenix trace as span annotations.

**2. Offline — batch, reference-based.** Run against the curated QA set for a fuller regression check, including retrieval-specific metrics that need a ground-truth reference:

```bash
python -m eval.generate_predictions   # run the real pipeline over eval/qa_set.json -> predictions.json
python -m eval.score_predictions      # score predictions.json -> ragas_results.csv
```

This scores every answerable question on `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall`, prints per-question and aggregate results, and writes `eval/ragas_results.csv`. Out-of-scope questions in the QA set are graded separately with a simple refusal check instead of an LLM judge.

---

## Tech stack

Python · LangGraph · OpenAI-compatible chat + embeddings API · Weaviate (hybrid search) · Redis (parent-chunk store) · Arize Phoenix (tracing) · Ragas (evaluation) · Pydantic v2 (settings) · pandas