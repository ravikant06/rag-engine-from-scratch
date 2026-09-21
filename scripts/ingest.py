"""
Ingestion: docs/ -> chunks -> embeddings -> Qdrant

Run:
  python scripts/ingest.py                  # full ingest (embeds + upserts)
  python scripts/ingest.py --payload-only   # refresh metadata, no embedding calls
  python scripts/ingest.py --tenant acme    # ingest under a different tenant
"""
import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Allow `from src import ...` when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chunker, config, embeddings, loader, vector_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs/ into Qdrant.")
    parser.add_argument("--tenant", default=config.TENANT_ID,
                        help=f"tenant to stamp on every chunk (default: {config.TENANT_ID})")
    parser.add_argument("--payload-only", action="store_true",
                        help="update metadata in place; skips embedding entirely")
    args = parser.parse_args()

    batch_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"

    print(f"Loading documents from {config.DOCS_DIR} ...")
    documents = loader.load_documents(config.DOCS_DIR)

    chunks = chunker.chunk_documents(
        documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP,
        tenant_id=args.tenant, batch_id=batch_id,
    )

    client = vector_store.get_client()

    if args.payload_only:
        # Chunk text is unchanged, so the existing vectors remain correct.
        vector_store.ensure_payload_indexes(client)
        written = vector_store.update_payloads(client, chunks)
        print(f"Payloads updated: {written} (no embedding calls made)")
    else:
        print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL} ...")
        vectors = embeddings.embed_documents(chunks)
        vector_store.ensure_collection(client)
        written = vector_store.upsert_chunks(client, chunks, vectors)
        print(f"Vectors indexed:  {written}")

    print()
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created:   {len(chunks)}")
    print(f"Tenant:           {args.tenant}")
    print(f"Batch id:         {batch_id}")
    print(f"Collection:       {config.COLLECTION_NAME} @ {config.QDRANT_URL}")


if __name__ == "__main__":
    main()
