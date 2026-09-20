"""
Ingestion: docs/ -> chunks -> embeddings -> Qdrant

Run:  python scripts/ingest.py
"""
import sys
from pathlib import Path

# Allow `from src import ...` when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chunker, config, embeddings, loader, vector_store  # noqa: E402


def main() -> None:
    print(f"Loading documents from {config.DOCS_DIR} ...")
    documents = loader.load_documents(config.DOCS_DIR)

    chunks = chunker.chunk_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL} ...")
    vectors = embeddings.embed_documents(chunks)

    client = vector_store.get_client()
    vector_store.ensure_collection(client)
    indexed = vector_store.upsert_chunks(client, chunks, vectors)

    print()
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created:   {len(chunks)}")
    print(f"Vectors indexed:  {indexed}")
    print(f"Collection:       {config.COLLECTION_NAME} @ {config.QDRANT_URL}")


if __name__ == "__main__":
    main()
