"""
Step 2 of the pipeline: split each document into overlapping chunks.

Why chunk at all?
  - Embedding models have input limits, and one vector for a whole document
    blurs many topics together. Smaller pieces give sharper, more
    retrievable vectors.

Why overlap?
  - A sentence that straddles a chunk boundary would otherwise be cut in
    half and lose meaning. Overlap repeats the tail of one chunk at the head
    of the next so no idea is split cleanly in two.

CHUNK_SIZE / CHUNK_OVERLAP are measured in characters. As a rough rule,
1 token ~ 4 characters of English, so 800 chars ~ 200 tokens.
"""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding window over the text. Tries to end each chunk at a newline
    or sentence boundary so chunks read naturally, but never exceeds chunk_size."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer to cut at a paragraph or sentence end within the last 20% of the window.
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > chunk_size * 0.8:
                end = start + cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap  # step back so neighbours share `overlap` chars
    return chunks


def chunk_documents(documents: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """
    Turn documents into chunk dicts. Each chunk keeps a pointer to its source
    so we can cite it later:
        {"chunk_id": "architecture-3", "source": "architecture.md",
         "doc_id": "architecture", "chunk_index": 3, "text": "..."}
    """
    all_chunks = []
    for doc in documents:
        for i, piece in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['id']}-{i}",
                    "doc_id": doc["id"],
                    "source": doc["source"],
                    "chunk_index": i,
                    "text": piece,
                }
            )
    return all_chunks
