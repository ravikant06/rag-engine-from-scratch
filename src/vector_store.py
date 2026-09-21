"""
Step 4 of the pipeline: store vectors in Qdrant and search them.

Each Qdrant "point" = one chunk:
    id      -> a UUID derived from chunk_id (Qdrant requires int or UUID ids)
    vector  -> the embedding
    payload -> chunk_id, doc_id, source, chunk_index, text, tenant_id,
               doc_type, heading, services, ...  (plain metadata)

Payload fields used in filters get an index. Without one Qdrant still
filters correctly, but by scanning rather than by lookup.
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src import config


def get_client() -> QdrantClient:
    client = QdrantClient(url=config.QDRANT_URL)
    try:
        client.get_collections()  # cheap call to confirm Qdrant is reachable
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach Qdrant at {config.QDRANT_URL}. Is Docker running?\n"
            "Start it with:  docker run -d --name qdrant -p 6333:6333 qdrant/qdrant"
        ) from exc
    return client


# Payload fields we filter on, and the index type each needs.
PAYLOAD_INDEXES = {
    "tenant_id": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,
    "services": models.PayloadSchemaType.KEYWORD,
    "severity": models.PayloadSchemaType.KEYWORD,
    "owner": models.PayloadSchemaType.KEYWORD,
    "date_ts": models.PayloadSchemaType.INTEGER,
    "ingested_ts": models.PayloadSchemaType.INTEGER,
}


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection if missing. Cosine distance suits normalised text embeddings."""
    if not client.collection_exists(config.COLLECTION_NAME):
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=config.EMBEDDING_DIM, distance=models.Distance.COSINE
            ),
        )
    ensure_payload_indexes(client)


def ensure_payload_indexes(client: QdrantClient) -> None:
    """Create a payload index per filterable field. Safe to call repeatedly."""
    for field_name, field_schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=config.COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception:
            pass  # already present


def _point_id(chunk_id: str) -> str:
    # Deterministic UUID: re-ingesting the same chunk overwrites instead of duplicating.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def upsert_chunks(client: QdrantClient, chunks: list[dict], vectors: list[list[float]]) -> int:
    points = [
        models.PointStruct(id=_point_id(c["chunk_id"]), vector=v, payload=c)
        for c, v in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=config.COLLECTION_NAME, points=points)
    return len(points)


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int,
    query_filter: models.Filter | None = None,
) -> list[dict]:
    """
    Return the top_k most similar chunks, each with its similarity score.

    `query_filter` is applied during the search (pre-filter), so top_k chunks
    come from the matching subset — not from an unfiltered top_k that is then
    trimmed down.
    """
    result = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return [{**hit.payload, "score": hit.score} for hit in result.points]


def update_payloads(client: QdrantClient, chunks: list[dict]) -> int:
    """
    Overwrite payloads in place, leaving vectors untouched.

    This is the cheap path for metadata-only changes: no embedding calls, no
    re-indexing. Valid only while chunk boundaries are unchanged, since the
    point id is derived from chunk_id.
    """
    for chunk in chunks:
        client.set_payload(
            collection_name=config.COLLECTION_NAME,
            payload=chunk,
            points=[_point_id(chunk["chunk_id"])],
        )
    return len(chunks)
