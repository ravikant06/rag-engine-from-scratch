"""
Step 3 of the pipeline: turn text into vectors with the Gemini embedding API.

Uses the official `google-genai` SDK (`from google import genai`).
"""
from google import genai
from google.genai import types

from src import config

# One shared client for the whole process.
_client = genai.Client(api_key=config.require_api_key())

# Gemini lets you tell the model what the embedding is for. Documents and
# queries get slightly different treatment, which improves retrieval.
_DOC_TASK = "RETRIEVAL_DOCUMENT"
_QUERY_TASK = "RETRIEVAL_QUERY"

_BATCH_SIZE = 100  # API limit on strings per embed_content call


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        try:
            response = _client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=config.EMBEDDING_DIM,
                ),
            )
        except Exception as exc:  # network / auth / quota problems
            raise SystemExit(f"Gemini embedding call failed: {exc}") from exc
        vectors.extend(e.values for e in response.embeddings)
    return vectors


def embed_text(text: str) -> list[float]:
    """Embed a single user question."""
    return _embed([text], _QUERY_TASK)[0]


def embed_documents(chunks: list[dict]) -> list[list[float]]:
    """Embed the `text` field of every chunk, preserving order."""
    return _embed([c["text"] for c in chunks], _DOC_TASK)
