"""
Ingest pipeline for the Compliance RAG Agent.

Loads a source document (PDF, TXT, MD, or CSV), extracts its content,
and writes the resulting parent/child chunks into the vector store.

Usage:
    python ingest_pipeline.py path/to/document.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app.services.ingest import ingest_pdf, ingest_text

SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".csv"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | {".pdf"}


def ingest_document(file_path: str | Path) -> dict[str, Any]:
    """Ingest a single document into the vector store.

    Args:
        file_path: Path to a PDF, TXT, MD, or CSV file.

    Returns:
        Ingestion stats containing at least ``filename``, ``parents_added``,
        and ``children_added``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        result = ingest_pdf(path.read_bytes(), path.name)
    elif suffix in SUPPORTED_TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        result = ingest_text(text, path.name)
    else:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {supported}")

    print(
        f"Ingested '{result['filename']}': "
        f"{result['parents_added']} parent chunk(s), "
        f"{result['children_added']} child chunk(s) added"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("Usage: python ingest_pipeline.py <path/to/document>")
        return 1

    try:
        ingest_document(argv[0])
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())