"""
Step 4 of the pipeline: store vectors in Qdrant and search them.

Each Qdrant "point" = one chunk:
    id      -> a UUID derived from chunk_id (Qdrant requires int or UUID ids)
    vector  -> the embedding
    payload -> chunk_id, doc_id, source, chunk_index, text  (plain metadata)
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


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection if missing. Cosine distance suits normalised text embeddings."""
    if client.collection_exists(config.COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=config.EMBEDDING_DIM, distance=models.Distance.COSINE
        ),
    )


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


def search(client: QdrantClient, query_vector: list[float], top_k: int) -> list[dict]:
    """Return the top_k most similar chunks, each with its similarity score."""
    result = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [{**hit.payload, "score": hit.score} for hit in result.points]
