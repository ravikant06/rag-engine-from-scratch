"""
Step 5b: agentic retrieval — let the model choose its own filters.

`scripts/ask.py` makes the caller supply filters (--doc-type, --service, ...).
A real user just asks a question, so here retrieval is exposed to the model as
a *tool*. The model fills in the filter arguments as part of the same call it
answers with, and can search again with different filters if the first attempt
comes back empty.

This module is the Client in the Adapter pattern: it talks only to
src.llm.LLMAdapter and holds no provider-specific code, so the same loop runs
against Gemini, OpenAI or Anthropic.

Two deliberate design choices:

  - `tenant_id` is NOT a tool parameter. It is injected server-side from
    config (in production: from the auth token). A model that could choose
    its own tenant would be a data-leak waiting to happen.

  - Every tool call is recorded in a trace and printed. A wrongly-inferred
    filter silently hides the right answer, so the filters must be visible
    to whoever is reading the output.
"""
from src import config, embeddings, filters, trace, vector_store
from src.llm import Message, ToolResult, ToolSpec, get_adapter

MAX_STEPS = 5  # search rounds per question, before we force an answer

SYSTEM_INSTRUCTION = """You answer questions about an engineering team's internal documentation.

You cannot see the documents directly. Use the `search_docs` tool to retrieve them.

Guidelines:
- Always call `search_docs` at least once before answering.
- Set filters only when the question clearly implies them. An unnecessary
  filter can hide the correct answer.
- If a filtered search returns nothing, search again with fewer filters
  before concluding the answer is absent.
- Answer ONLY from retrieved chunks. Never invent facts.
- If the documents do not contain the answer, say:
  "I could not find this in the documentation."
- End your answer with a "Sources:" line listing the filenames you used."""

# Declared once, in plain JSON Schema. Each adapter rewraps this in whatever
# envelope its provider expects — see src/llm/*.py.
SEARCH_DOCS = ToolSpec(
    name="search_docs",
    description=(
        "Semantic search over the engineering documentation. "
        "Returns the most relevant chunks with their source filenames."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to search for, as a natural-language phrase. Rephrase "
                    "the user's question into the wording you would expect the "
                    "documentation to use."
                ),
            },
            "doc_type": {
                "type": "string",
                "enum": ["guide", "reference", "incident", "runbook", "onboarding"],
                "description": "Restrict to one class of document.",
            },
            "service": {
                "type": "string",
                "enum": list(config.KNOWN_SERVICES),
                "description": "Restrict to chunks mentioning this service.",
            },
            "source": {
                "type": "string",
                "description": "Restrict to one file, e.g. 'api.md'.",
            },
            "severity": {
                "type": "string",
                "description": "Incident severity, e.g. 'SEV-2'.",
            },
            "date_from": {
                "type": "string",
                "description": "YYYY-MM-DD. Only dated documents (incidents) have dates.",
            },
            "date_to": {
                "type": "string",
                "description": "YYYY-MM-DD. Only dated documents (incidents) have dates.",
            },
        },
        "required": ["query"],
    },
)


def _run_search(args: dict, tenant_id: str | None, top_k: int) -> list[dict]:
    """Execute one tool call. tenant_id comes from us, never from the model."""
    query = args.get("query", "")
    where = {k: v for k, v in args.items() if k != "query" and v}

    client = vector_store.get_client()
    if not client.collection_exists(config.COLLECTION_NAME):
        raise SystemExit("Collection is empty. Run `python scripts/ingest.py` first.")

    if trace.is_on():
        trace.section("TOOL search_docs")
        trace.kv("query", repr(query))
        trace.kv("model filters", trace.compact_json(where) if where else "(none)")
        trace.kv("tenant injected", tenant_id)

    query_vector = embeddings.embed_text(query)
    query_filter = filters.build_filter(tenant_id=tenant_id, **where)
    return vector_store.search(client, query_vector, top_k, query_filter=query_filter)


def _tool_payload(chunks: list[dict]) -> dict:
    """What the model sees back. Deliberately trimmed to what it needs to cite."""
    return {
        "results": [
            {
                "source": c["source"],
                "heading": c.get("heading"),
                "score": round(c["score"], 3),
                "text": c["text"],
            }
            for c in chunks
        ],
        "count": len(chunks),
    }


def answer(
    question: str,
    top_k: int = config.TOP_K,
    tenant_id: str | None = None,
    llm=None,
) -> tuple[list[dict], str, list[dict]]:
    """
    Agentic RAG. Returns (chunks_seen, final_answer, trace).

    `llm` is injected for testability and provider choice; it defaults to
    whatever config.LLM_PROVIDER selects.
    """
    llm = llm or get_adapter()
    tenant_id = tenant_id or config.TENANT_ID

    trace.reset()
    trace.section("QUERY RECEIVED")
    trace.kv("question", repr(question))
    trace.kv("mode", "agentic (model chooses filters)")
    trace.kv("tenant", tenant_id)
    trace.kv("top_k", top_k)
    trace.kv("provider", getattr(llm, "provider", "?"))
    trace.kv("max search rounds", MAX_STEPS)

    messages: list[Message] = [Message.user(question)]
    seen: dict[str, dict] = {}   # chunk_id -> chunk, deduped across searches
    steps: list[dict] = []

    for _ in range(MAX_STEPS):
        reply = llm.complete(messages, tools=[SEARCH_DOCS], system=SYSTEM_INSTRUCTION)

        if not reply.wants_tools:
            answer_text = reply.text or "(empty response from model)"
            trace.section("DONE")
            trace.kv("searches run", len(steps))
            trace.kv("unique chunks seen", len(seen))
            return list(seen.values()), answer_text, steps

        messages.append(Message.assistant(text=reply.text, tool_calls=reply.tool_calls))

        for call in reply.tool_calls:
            chunks = _run_search(call.arguments, tenant_id, top_k)
            for chunk in chunks:
                seen.setdefault(chunk["chunk_id"], chunk)
            steps.append(
                {
                    "query": call.arguments.get("query", ""),
                    "where": {k: v for k, v in call.arguments.items() if k != "query" and v},
                    "count": len(chunks),
                }
            )
            messages.append(
                Message.tool(
                    ToolResult(id=call.id, name=call.name, content=_tool_payload(chunks))
                )
            )

    # Budget exhausted. Ask once more with no tools available, so the model has
    # to answer from what it already retrieved instead of searching forever.
    trace.section("BUDGET EXHAUSTED - forcing an answer (no tools offered)")
    final = llm.complete(
        messages,
        system=(
            SYSTEM_INSTRUCTION
            + "\n\nYou have no more searches left. Answer from the results you "
            "already have, or say you could not find it."
        ),
    )
    return (
        list(seen.values()),
        final.text or "I could not find this in the documentation.",
        steps,
    )
