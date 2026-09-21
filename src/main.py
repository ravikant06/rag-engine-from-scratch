"""
Interactive REPL. Ask questions in plain language; the model picks its own
retrieval filters via the search_docs tool (see src/agent.py).

Run with:  python -m src.main
           python -m src.main --plain     # no tool calling, filters unused
"""
import argparse

from src import agent, config, rag


def print_trace(trace: list[dict]) -> None:
    """Show which searches the model chose, so inferred filters are visible."""
    if not trace:
        return
    print("SEARCHES:")
    for i, step in enumerate(trace, 1):
        where = ", ".join(f"{k}={v}" for k, v in sorted(step["where"].items()))
        print(f"  [{i}] query={step['query']!r}")
        print(f"      filters: tenant={config.TENANT_ID}{', ' + where if where else ''}")
        print(f"      hits:    {step['count']}")
    print()


def print_result(question: str, chunks: list[dict], final_answer: str) -> None:
    print(f"\nQUESTION: {question}\n")
    print("RETRIEVED CHUNKS:")
    for i, c in enumerate(chunks, 1):
        preview = c["text"].replace("\n", " ")[:160]
        where = c["source"]
        if c.get("heading"):
            where += f" > {c['heading']}"
        print(f"  [{i}] {where} (chunk {c['chunk_index']}, score={c['score']:.3f})")
        print(f"      {preview}...")
    print("\nANSWER:")
    print(final_answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive RAG REPL.")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="skip tool calling; retrieve once with no filters",
    )
    args = parser.parse_args()

    mode = "plain retrieval" if args.plain else "tool-calling retrieval"
    print(f"RAG REPL ({mode}, tenant={config.TENANT_ID}) - ask a question, or 'exit' to quit.")

    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break

        if args.plain:
            chunks, final_answer = rag.answer(question)
            print_result(question, chunks, final_answer)
        else:
            chunks, final_answer, trace = agent.answer(question)
            print()
            print_trace(trace)
            print_result(question, chunks, final_answer)


if __name__ == "__main__":
    main()
