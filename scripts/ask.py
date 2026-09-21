"""
Ask one question against the indexed docs, optionally narrowed by metadata.

Run:
  python scripts/ask.py "How is payment-service deployed?"
  python scripts/ask.py "What broke?" --doc-type incident
  python scripts/ask.py "How do we roll back?" --service payment-service
  python scripts/ask.py "Any SEV-2s?" --doc-type incident --date-from 2026-03-01
  python scripts/ask.py "Rate limits?" --source api.md --tenant acme
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, filters, rag, trace  # noqa: E402
from src.main import print_result  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question against the indexed documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", nargs="+", help="the question to ask")
    parser.add_argument(
        "--tenant",
        default=config.TENANT_ID,
        help=f"tenant to search within (default: {config.TENANT_ID})",
    )
    parser.add_argument("--top-k", type=int, default=config.TOP_K,
                        help=f"chunks to retrieve (default: {config.TOP_K})")
    parser.add_argument("--doc-type", action="append",
                        help="guide | reference | incident | runbook | onboarding (repeatable)")
    parser.add_argument("--source", action="append", help="filename, e.g. api.md (repeatable)")
    parser.add_argument("--doc-id", action="append", help="document id, e.g. deployment")
    parser.add_argument("--service", action="append",
                        help=f"one of: {', '.join(config.KNOWN_SERVICES)} (repeatable)")
    parser.add_argument("--severity", action="append", help="e.g. SEV-2")
    parser.add_argument("--owner", action="append", help="owning team")
    parser.add_argument("--date-from", help="YYYY-MM-DD (dated docs only)")
    parser.add_argument("--date-to", help="YYYY-MM-DD (dated docs only)")
    parser.add_argument("-t", "--trace", action="store_true",
                        help="print every step: prompts, embeddings, filters, results")
    parser.add_argument("--trace-full", action="store_true",
                        help="like --trace, without truncating long prompts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trace or args.trace_full:
        trace.enable(True, full=args.trace_full)
    question = " ".join(args.question)

    # Only non-empty filters are passed on; everything else stays unrestricted.
    where = {
        key: value
        for key, value in {
            "doc_type": args.doc_type,
            "source": args.source,
            "doc_id": args.doc_id,
            "service": args.service,
            "severity": args.severity,
            "owner": args.owner,
            "date_from": args.date_from,
            "date_to": args.date_to,
        }.items()
        if value
    }

    print(f"FILTERS: {filters.describe(tenant_id=args.tenant, **where)}")
    chunks, final_answer = rag.answer(
        question, top_k=args.top_k, tenant_id=args.tenant, where=where
    )
    print_result(question, chunks, final_answer)


if __name__ == "__main__":
    main()
