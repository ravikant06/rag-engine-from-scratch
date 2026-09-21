"""
Step 1 of the pipeline: read raw documents from disk.

A "document" here is just a dict — the pipeline needs no richer type.

Alongside the text we extract document-level metadata that is already present
in the files (title, doc_type, and the Date/Severity/Owner fields incident
reports carry). None of it changes `text`, so none of it affects embeddings.
"""
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from src import config

SUPPORTED_EXTENSIONS = {".md", ".txt"}

_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FIELD_RE = re.compile(r"^\*\*(Date|Severity|Owner|Duration):\*\*\s*(.+?)\s*$", re.MULTILINE)
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _doc_type(stem: str) -> str:
    """Map a filename to a coarse document class using config.DOC_TYPE_RULES."""
    lowered = stem.lower()
    for prefix, doc_type in config.DOC_TYPE_RULES:
        if lowered.startswith(prefix):
            return doc_type
    return config.DEFAULT_DOC_TYPE


def _to_epoch_day(value: str) -> int | None:
    """'2026-03-14' -> epoch seconds, so Qdrant can do numeric range filters."""
    match = _ISO_DATE_RE.search(value)
    if not match:
        return None
    dt = datetime.strptime(match.group(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def extract_doc_metadata(stem: str, text: str) -> dict:
    """
    Pull the structured fields the documents already contain.

    Returns only keys that were actually found, so chunks never carry
    empty placeholders.
    """
    meta: dict = {"doc_type": _doc_type(stem)}

    title = _H1_RE.search(text)
    if title:
        meta["title"] = title.group(1).strip()

    for name, raw in _FIELD_RE.findall(text):
        key = name.lower()
        if key == "date":
            meta["date"] = raw
            epoch = _to_epoch_day(raw)
            if epoch is not None:
                meta["date_ts"] = epoch
        elif key == "severity":
            meta["severity"] = raw.upper()
        elif key == "owner":
            meta["owner"] = raw
        # Duration is captured for completeness but not indexed.

    return meta


def load_documents(docs_dir: Path) -> list[dict]:
    """
    Return one dict per file:
        {
          "id": "architecture",          # filename without extension
          "source": "architecture.md",   # shown to the user in answers
          "text": "...full file text...",
          "metadata": {"doc_type": "guide", "title": "...", ...}
        }
    """
    if not docs_dir.exists():
        raise SystemExit(f"Docs directory not found: {docs_dir}")

    ingested_at = datetime.now(timezone.utc)
    documents = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  skipping empty file: {path.name}")
            continue

        metadata = extract_doc_metadata(path.stem, text)
        metadata.update(
            {
                "path": str(path),
                "chars": len(text),
                # Lets a later ingest skip documents that have not changed.
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                "ingested_at": ingested_at.isoformat(timespec="seconds"),
                "ingested_ts": int(ingested_at.timestamp()),
            }
        )

        documents.append(
            {
                "id": path.stem,
                "source": path.name,
                "text": text,
                "metadata": metadata,
            }
        )

    if not documents:
        raise SystemExit(f"No .md or .txt files with content found in {docs_dir}")
    return documents
