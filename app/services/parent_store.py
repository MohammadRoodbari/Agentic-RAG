import json
import uuid
from typing import Dict, List, Optional

import redis

from app.config import settings


def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def make_parent_id(filename: str, page: int, parent_index: int) -> str:
    """Deterministic UUID so re-ingesting the same document overwrites
    instead of duplicating. Same idea as the sha256-based chunk ids in
    the original ingest.py, just UUID-shaped so Qdrant/Redis accept it."""
    key = f"{filename}:{page}:{parent_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def save_parent(
    parent_id: str,
    text: str,
    source: str,
    page: int,
    parent_index: int,
) -> None:
    client = get_redis_client()
    payload = {
        "text": text,
        "source": source,
        "page": page,
        "parent_index": parent_index,
    }
    key = f"parent:{parent_id}"
    client.set(key, json.dumps(payload))
    if settings.redis_parent_ttl_seconds:
        client.expire(key, settings.redis_parent_ttl_seconds)
    # Track membership per source so we can delete cleanly on re-ingest
    # without scanning + JSON-parsing every key in Redis.
    client.sadd(f"parent_ids_by_source:{source}", parent_id)


def get_parents(parent_ids: List[str]) -> Dict[str, Optional[dict]]:
    """Batch fetch parents by id. Missing ids map to None so callers can
    skip orphaned references (e.g. a TTL-evicted parent) instead of
    crashing on a KeyError."""
    if not parent_ids:
        return {}
    client = get_redis_client()
    keys = [f"parent:{pid}" for pid in parent_ids]
    raw_values = client.mget(keys)
    return {
        pid: (json.loads(raw) if raw is not None else None)
        for pid, raw in zip(parent_ids, raw_values)
    }


def delete_parents_by_source(source: str) -> int:
    """Delete all parent docs for a given source filename. Called before
    re-ingesting a file so old parents don't linger after their child
    chunks have been replaced in Qdrant."""
    client = get_redis_client()
    set_key = f"parent_ids_by_source:{source}"
    parent_ids = client.smembers(set_key)
    if not parent_ids:
        return 0
    keys = [f"parent:{pid}" for pid in parent_ids]
    deleted = client.delete(*keys)
    client.delete(set_key)
    return deleted
