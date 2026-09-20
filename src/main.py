"""
Tiny interactive REPL: keeps asking questions until you type 'exit'.
Run with:  python -m src.main
"""
from src import rag


def print_result(question: str, chunks: list[dict], final_answer: str) -> None:
    print(f"\nQUESTION: {question}\n")
    print("RETRIEVED CHUNKS:")
    for i, c in enumerate(chunks, 1):
        preview = c["text"].replace("\n", " ")[:160]
        print(f"  [{i}] {c['source']} (chunk {c['chunk_index']}, score={c['score']:.3f})")
        print(f"      {preview}...")
    print("\nANSWER:")
    print(final_answer)


def main() -> None:
    print("RAG REPL - type a question, or 'exit' to quit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        chunks, final_answer = rag.answer(question)
        print_result(question, chunks, final_answer)


if __name__ == "__main__":
    main()
