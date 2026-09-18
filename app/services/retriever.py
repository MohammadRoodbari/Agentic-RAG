from typing import Dict, List, Optional

from weaviate.classes.query import HybridFusion, MetadataQuery

from app.agent.state import RetrievedChunk
from app.config import settings
from app.services.ingest import embed_texts, get_weaviate_client
from app.services.parent_store import get_parents


def hybrid_retrieve(query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    if top_k is None:
        top_k = settings.final_top_k
    retrieval_k = settings.retrieval_top_k

    with get_weaviate_client() as client:
        if not client.collections.exists(settings.weaviate_collection):
            return []
        collection = client.collections.get(settings.weaviate_collection)
        if collection.aggregate.over_all(total_count=True).total_count == 0:
            return []

        query_vector = embed_texts([query])[0]

        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=settings.hybrid_alpha,
            fusion_type=HybridFusion.RANKED,
            limit=retrieval_k,
            return_metadata=MetadataQuery(score=True),
        )
        # Collapse fused CHILD hits to unique PARENTS -- several child
        # chunks hitting on the same parent shouldn't count as separate
        # results, and the parent (not the child fragment) is what we
        # want to hand back as context.
        parent_order: List[str] = []
        parent_best_score: Dict[str, float] = {}
        seen_parents = set()
        for obj in response.objects:
            parent_id = obj.properties["parent_id"]
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_order.append(parent_id)
                parent_best_score[parent_id] = obj.metadata.score
            if len(parent_order) >= top_k:
                break

    parents = get_parents(parent_order)
    results: List[RetrievedChunk] = []
    for parent_id in parent_order:
        parent = parents.get(parent_id)
        if parent is None:
            # Orphaned reference -- child chunk points at a parent that
            # no longer exists in Redis (e.g. evicted by TTL). Skip it
            # rather than raising.
            continue
        results.append(
            RetrievedChunk(
                text=parent["text"],
                source=parent["source"],
                page=parent["page"],
                chunk_index=parent["parent_index"],
                score=parent_best_score[parent_id],
            )
        )

    return results
