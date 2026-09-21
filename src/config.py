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

# Which LLM adapter to use: gemini | openai | anthropic (see src/llm/).
# Swapping this also means setting a matching GENERATION_MODEL and API key.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "engineering_docs")

# --- Multi-tenancy ---
# Every chunk is stamped with this at ingest time, and every search is scoped
# to it. One tenant can never retrieve another tenant's chunks.
TENANT_ID = os.getenv("TENANT_ID", "default")

# --- Metadata extraction ---
# Service names we recognise in chunk text. Kept explicit rather than regexed
# so "cross-service" and similar words do not become false positives.
KNOWN_SERVICES = [
    "payment-service",
    "order-service",
    "inventory-service",
    "notification-service",
    "api-gateway",
]

# Filename prefix -> doc_type. First match wins; anything unmatched is "guide".
DOC_TYPE_RULES = [
    ("incident", "incident"),
    ("onboarding", "onboarding"),
    ("troubleshooting", "runbook"),
    ("api", "reference"),
    ("database", "reference"),
]
DEFAULT_DOC_TYPE = "guide"

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
