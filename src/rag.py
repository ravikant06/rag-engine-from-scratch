"""
Steps 5-7 of the pipeline: retrieve -> build prompt -> generate.

This is the heart of RAG. Read it top to bottom.
"""
from google import genai

from src import config, embeddings, vector_store

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


def retrieve(question: str, top_k: int = config.TOP_K) -> list[dict]:
    """question -> embedding -> Qdrant similarity search -> top_k chunks"""
    client = vector_store.get_client()
    if not client.collection_exists(config.COLLECTION_NAME):
        raise SystemExit("Collection is empty. Run `python scripts/ingest.py` first.")
    query_vector = embeddings.embed_text(question)
    return vector_store.search(client, query_vector, top_k)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Stitch retrieved chunks into the prompt, each labelled with its source."""
    context_blocks = [
        f"[{i + 1}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks)
    ]
    return PROMPT_TEMPLATE.format(context="\n\n".join(context_blocks), question=question)


def generate(prompt: str) -> str:
    try:
        response = _client.models.generate_content(
            model=config.GENERATION_MODEL, contents=prompt
        )
    except Exception as exc:
        raise SystemExit(f"Gemini generation call failed: {exc}") from exc
    return response.text or "(empty response from model)"


def answer(question: str, top_k: int = config.TOP_K) -> tuple[list[dict], str]:
    """Full RAG flow. Returns (retrieved_chunks, final_answer) so callers can show both."""
    chunks = retrieve(question, top_k)
    prompt = build_prompt(question, chunks)
    return chunks, generate(prompt)
