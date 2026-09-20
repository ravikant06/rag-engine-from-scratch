"""
Central configuration. Every tunable value lives here and can be overridden
with an environment variable (or a .env file in the project root).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root so `python scripts/ask.py` works from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
# gemini-embedding-001 supports 768 / 1536 / 3072 dims (Matryoshka). 768 is
# plenty for a small corpus and keeps Qdrant small and fast.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash")

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "engineering_docs")

# --- RAG knobs ---
DOCS_DIR = PROJECT_ROOT / "docs"
TOP_K = int(os.getenv("TOP_K", "4"))                     # how many chunks to retrieve
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))         # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))   # characters shared between neighbours


def require_api_key() -> str:
    """Fail fast with a clear message instead of a cryptic 401 later."""
    if not GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Add it to .env (see .env.example) "
            "or export it in your shell."
        )
    return GEMINI_API_KEY
