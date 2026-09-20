"""
Ask one question against the indexed docs.

Run:  python scripts/ask.py "How is payment-service deployed?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rag  # noqa: E402
from src.main import print_result  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts/ask.py "your question here"')
    question = " ".join(sys.argv[1:])
    chunks, final_answer = rag.answer(question)
    print_result(question, chunks, final_answer)


if __name__ == "__main__":
    main()
