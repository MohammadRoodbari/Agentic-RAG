import uuid
from typing import List, Tuple

import weaviate
from openai import OpenAI
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.parent_store import (
    delete_parents_by_source,
    make_parent_id,
    save_parent,
)
from app.services.text_extractor import extract_text_from_pdf


def get_weaviate_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host=settings.weaviate_http_host,
        port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
    )


def ensure_collection() -> None:
    with get_weaviate_client() as client:
        if client.collections.exists(settings.weaviate_collection):
            return
        client.collections.create(
            name=settings.weaviate_collection,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(
                    name="source",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(
                    name="parent_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="page", data_type=DataType.INT),
                Property(name="parent_index", data_type=DataType.INT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )


def embed_texts(texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=settings.require_api_key(),base_url=settings.base_url)
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_pdf(pdf_bytes: bytes, filename: str) -> dict:
    pages = extract_text_from_pdf(pdf_bytes, max_pages=settings.max_pdf_pages)
    return _ingest_pages(pages, filename)


def ingest_text(text: str, filename: str) -> dict:
    return _ingest_pages([(1, text)], filename)


def _ingest_pages(pages: List[Tuple[int, str]], filename: str) -> dict:
    ensure_collection()

    _delete_existing_source(filename)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_texts: list[str] = []
    child_ids: list[str] = []
    child_properties: list[dict] = []
    parents_added = 0

    for page_num, page_text in pages:
        parent_chunks = parent_splitter.split_text(page_text)

        for parent_index, parent_text in enumerate(parent_chunks):
            parent_id = make_parent_id(filename, page_num, parent_index)
            save_parent(
                parent_id=parent_id,
                text=parent_text,
                source=filename,
                page=page_num,
                parent_index=parent_index,
            )
            parents_added += 1

            for chunk_index, child_text in enumerate(
                child_splitter.split_text(parent_text)
            ):
                child_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"{parent_id}:{chunk_index}")
                )
                child_texts.append(child_text)
                child_ids.append(child_id)
                child_properties.append({
                    "text": child_text,
                    "parent_id": parent_id,
                    "source": filename,
                    "page": page_num,
                    "parent_index": parent_index,
                    "chunk_index": chunk_index,
                })

    if not child_texts:
        return {"filename": filename, "parents_added": 0, "children_added": 0}

    dense_embeddings = embed_texts(child_texts)

    with get_weaviate_client() as client:
        collection = client.collections.get(settings.weaviate_collection)
        with collection.batch.fixed_size(batch_size=256) as batch:
            for i in range(len(child_ids)):
                batch.add_object(
                    uuid=child_ids[i],
                    vector=dense_embeddings[i],
                    properties=child_properties[i],
                )

        failed = collection.batch.failed_objects
        if failed:
            raise RuntimeError(
                f"{len(failed)} objects failed to insert for "
                f"'{filename}' (first error: {failed[0].message})"
            )

    return {
        "filename": filename,
        "parents_added": parents_added,
        "children_added": len(child_ids),
    }


def _delete_existing_source(filename: str) -> None:
    with get_weaviate_client() as client:
        if not client.collections.exists(settings.weaviate_collection):
            delete_parents_by_source(filename)
            return
        collection = client.collections.get(settings.weaviate_collection)
        collection.data.delete_many(
            where=Filter.by_property("source").equal(filename)
        )
    delete_parents_by_source(filename)
