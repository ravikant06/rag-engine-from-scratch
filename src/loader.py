"""
Step 1 of the pipeline: read raw documents from disk.

A "document" here is just a dict — the pipeline needs no richer type.
"""
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(docs_dir: Path) -> list[dict]:
    """
    Return one dict per file:
        {
          "id": "architecture",          # filename without extension
          "source": "architecture.md",   # shown to the user in answers
          "text": "...full file text...",
          "metadata": {"path": "...", "chars": 1234}
        }
    """
    if not docs_dir.exists():
        raise SystemExit(f"Docs directory not found: {docs_dir}")

    documents = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  skipping empty file: {path.name}")
            continue
        documents.append(
            {
                "id": path.stem,
                "source": path.name,
                "text": text,
                "metadata": {"path": str(path), "chars": len(text)},
            }
        )

    if not documents:
        raise SystemExit(f"No .md or .txt files with content found in {docs_dir}")
    return documents
