"""
Steps 5-7 of the pipeline: retrieve -> build prompt -> generate.

This is the heart of RAG. Read it top to bottom.
"""
from google import genai

from src import config, embeddings, filters, vector_store

_client = genai.Client(api_key=config.require_api_key())

PROMPT_TEMPLATE = """You are an assistant answering questions about an engineering team's internal documentation.

Rules:
- Answer ONLY using the context below.
- Do not invent facts. If the context does not contain the answer, say:
  "I could not find this in the documentation."
- End your answer with a line "Sources:" listing the filenames you actually used.

Context:
{context}

Question: {question}

Answer:"""


def retrieve(
    question: str,
    top_k: int = config.TOP_K,
    tenant_id: str | None = None,
    where: dict | None = None,
) -> list[dict]:
    """
    question -> embedding -> filtered Qdrant similarity search -> top_k chunks

    `tenant_id` defaults to config.TENANT_ID and is always applied.
    `where` is an optional dict of filters (doc_type, source, service,
    severity, owner, doc_id, date_from, date_to) — see src/filters.py.
    """
    client = vector_store.get_client()
    if not client.collection_exists(config.COLLECTION_NAME):
        raise SystemExit("Collection is empty. Run `python scripts/ingest.py` first.")
    query_vector = embeddings.embed_text(question)
    query_filter = filters.build_filter(tenant_id=tenant_id, **(where or {}))
    return vector_store.search(client, query_vector, top_k, query_filter=query_filter)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Stitch retrieved chunks into the prompt, each labelled with its source."""
    context_blocks = []
    for i, c in enumerate(chunks):
        label = c["source"]
        if c.get("heading"):
            # Restores the section title a chunk may have been cut away from.
            label += f" > {c['heading']}"
        context_blocks.append(f"[{i + 1}] (source: {label})\n{c['text']}")
    return PROMPT_TEMPLATE.format(context="\n\n".join(context_blocks), question=question)


def generate(prompt: str) -> str:
    try:
        response = _client.models.generate_content(
            model=config.GENERATION_MODEL, contents=prompt
        )
    except Exception as exc:
        raise SystemExit(f"Gemini generation call failed: {exc}") from exc
    return response.text or "(empty response from model)"


def answer(
    question: str,
    top_k: int = config.TOP_K,
    tenant_id: str | None = None,
    where: dict | None = None,
) -> tuple[list[dict], str]:
    """Full RAG flow. Returns (retrieved_chunks, final_answer) so callers can show both."""
    chunks = retrieve(question, top_k, tenant_id=tenant_id, where=where)
    if not chunks:
        # Filters can legitimately match nothing. Say so rather than asking the
        # model to answer from an empty context, which invites invention.
        return [], (
            "No documents matched the given filters, so I could not find this "
            "in the documentation."
        )
    prompt = build_prompt(question, chunks)
    return chunks, generate(prompt)
