# Design Document — RAG Engine from Scratch

Design decisions, their rationale and their tradeoffs. The README covers what
the system does and how to run it; this document covers *why it is built this
way* and what it deliberately does not do.

---

## 1. Goals and non-goals

**Goals**

- Every stage of RAG — chunk, embed, store, retrieve, prompt, generate — is a
  small readable module, not a framework call.
- Retrieval quality is inspectable: scores and sources are printed with every
  answer.
- Metadata is first-class, so filtering and lifecycle operations are possible.
- Provider-independent at the LLM boundary.

**Non-goals**

- Scale. The corpus is 7 documents / 23 chunks. Nothing here is tuned for
  millions of vectors, and several choices would be wrong at that size.
- Framework parity. No agent framework, no retriever abstraction beyond what
  the code needs.
- Production operations. No tracing, caching, rate limiting, or cost budgets.

---

## 2. Module map

```
src/
├── config.py        settings, read from env / .env
├── loader.py        docs/ -> documents + document-level metadata
├── chunker.py       documents -> chunks + chunk-level metadata
├── embeddings.py    text -> vectors (Gemini embedding API)
├── vector_store.py  Qdrant: collection, payload indexes, upsert, search
├── filters.py       user parameters -> Qdrant filter
├── rag.py           fixed pipeline: retrieve -> prompt -> generate
├── agent.py         agentic pipeline: model chooses filters via a tool
├── main.py          interactive REPL
└── llm/             provider-neutral LLM access (Adapter pattern)
    ├── types.py     Message, ToolCall, ToolResult, ToolSpec, LLMResponse
    ├── base.py      LLMAdapter — the Target interface
    ├── registry.py  @register + get_adapter() factory
    ├── gemini.py    adapter over google-genai
    ├── openai.py    adapter over openai
    └── anthropic.py adapter over anthropic

scripts/ingest.py    docs -> Qdrant (full, or --payload-only)
scripts/ask.py       one question with explicit filters
tests/test_adapter.py  drives the agent loop with a FakeAdapter
```

Dependency direction is one-way: `agent.py` and `rag.py` depend on
`vector_store`, `embeddings`, `filters` and `llm`. Nothing depends back on them.

---

## 3. Data flow

### Ingest

```
docs/*.md ──▶ loader.load_documents()
                 ├─ doc_type from filename rules
                 ├─ title from the H1
                 ├─ date / severity / owner from **Field:** lines
                 └─ content_hash, ingested_at
              ▼
           chunker.chunk_documents()
                 ├─ _chunk_spans() — sliding window, (start, end) offsets
                 ├─ heading + heading_path from the nearest heading above start
                 ├─ services matched against config.KNOWN_SERVICES
                 └─ tenant_id, batch_id, char_count
              ▼
           embeddings.embed_documents()   task_type=RETRIEVAL_DOCUMENT
              ▼
           vector_store.upsert_chunks()   point id = uuid5(chunk_id)
```

### Query — fixed pipeline (`scripts/ask.py`)

```
question + explicit filters
  ▼ embeddings.embed_text()            task_type=RETRIEVAL_QUERY
  ▼ filters.build_filter()             tenant always applied
  ▼ vector_store.search()              pre-filtered ANN
  ▼ rag.build_prompt()                 chunks labelled source > heading
  ▼ rag.generate()                     one LLM call
```

### Query — agentic pipeline (`python -m src.main`)

```
question
  ▼ llm.complete(messages, tools=[SEARCH_DOCS])
  ◀ tool_calls: search_docs(query=..., doc_type=...)
  ▼ _run_search()  ─ tenant injected here ─▶ embed ─▶ Qdrant
  ▼ append tool result to messages, loop
  ◀ text answer, or budget exhausted -> final call with no tools
```

---

## 4. Design decisions

### D1 — Character-based chunking with soft boundaries

`chunk_size=800`, `overlap=150`, cutting at a paragraph or sentence end within
the last 20% of the window.

*Rationale.* Predictable, dependency-free, and easy to reason about when
debugging retrieval.

*Tradeoff.* Not structure-aware: a Markdown table or a numbered runbook can be
split across chunks. Structure-aware splitting is the obvious upgrade and would
force a full re-embed.

### D2 — Metadata lives in the payload, never in the embedded text

Chunk metadata (heading, services, doc_type, dates) is stored in the Qdrant
payload. It is never prepended to `text` before embedding.

*Rationale.* Vectors depend only on `text`, so metadata can be added, corrected
or extended with `scripts/ingest.py --payload-only` — no embedding calls, no
re-indexing. When the chunker was refactored to extract headings, the 23 chunk
texts were diffed against a pre-change baseline to confirm boundaries had not
moved and existing vectors stayed valid.

*Tradeoff.* The embedding cannot benefit from the heading. Prepending the
heading to the text before embedding ("contextual retrieval") usually improves
recall, but requires a full re-embed and is therefore a separate experiment.

### D3 — Tenant isolation is structural, not a parameter

`tenant_id` is always applied in `filters.build_filter()`. A caller that omits
it gets `config.TENANT_ID` — never an unscoped search. It is deliberately
absent from the `search_docs` tool schema, and injected inside `_run_search()`.

*Rationale.* Narrowing filters are the model's to choose; isolation filters are
not. A model that could select its own tenant is a data-leak waiting to happen.
In a real deployment `config.TENANT_ID` would be replaced by a value read from
the authenticated session, never from user input.

### D4 — Filters are applied during search, not after

Qdrant receives `query_filter`, so `top_k` chunks come from the matching
subset. Post-filtering an unfiltered top-k would usually leave far fewer than
`top_k` results.

*Tradeoff.* Filtering increases hallucination risk when the filter is wrong: a
confidently-retrieved but irrelevant set looks identical to a relevant one. The
mitigation is visibility — every applied filter is printed.

### D5 — Payload indexes on every filterable field

`vector_store.PAYLOAD_INDEXES` declares nine fields; `ensure_payload_indexes()`
is idempotent and runs on every ingest.

*Rationale.* Correct without them, but scanned rather than looked up. At 23
chunks this is unmeasurable; declaring it keeps the schema honest about intent.

### D6 — Deterministic point ids

`uuid5(NAMESPACE_URL, chunk_id)` means re-ingesting overwrites rather than
duplicating, and `--payload-only` can address existing points.

*Tradeoff.* Only safe while chunk boundaries are stable. If a document shrinks,
chunks above the new count become orphans — see G1.

### D7 — Two query paths, kept separate

`rag.answer()` (fixed) and `agent.answer()` (agentic) both exist.

*Rationale.* The fixed path is what you use to *measure* retrieval, because
filters are held constant. The agentic path is what a user talks to. Measuring
against a pipeline that changes its own filters would not isolate the variable.

### D8 — Agentic retrieval with a bounded budget

The model chooses filters via the `search_docs` tool and may search again.
`MAX_STEPS = 5`; on exhaustion a final call is made **with the tool removed**,
forcing an answer or a refusal.

*Rationale.* The value is recovery: a zero-hit filter is visible to the model,
which can drop it and retry. A one-shot extract-then-search pipeline cannot.

*Tradeoff.* Measured on the SEV-1 question: 5 LLM calls and 4 embedding +
Qdrant round trips versus 1 each for the fixed path — roughly 4–5× cost and
latency, and *variable* per request. Production systems usually route simple
queries to the fixed path and reserve the loop for queries that need it.

*Observed quality issue.* The model sometimes spends its whole budget
rephrasing rather than concluding. A richer empty result (for example, listing
the severities actually present) would help it converge.

### D9 — Adapter pattern at the LLM boundary

```
   Client              Target (ABC)             Adaptee
 agent.py   ────────▶  LLMAdapter    ◀────────  google-genai
                       .complete()              openai
                            ▲                   anthropic
              Gemini / OpenAI / Anthropic adapters
                            │
                    registry.get_adapter()
```

Tool calling is one protocol with three envelopes. `agent.SEARCH_DOCS` is
declared once in plain JSON Schema; each adapter rewraps it.

| Provider | Tool envelope | Calls appear in |
|---|---|---|
| Gemini | `FunctionDeclaration(parameters=...)`, types uppercased | `parts[i].function_call` |
| OpenAI | `{"type":"function","function":{...}}` | `message.tool_calls` |
| Anthropic | `{"name","description","input_schema"}` | `tool_use` blocks |

*Principles applied.* Dependency Inversion — `agent.py` imports no provider
SDK. Open/Closed — a new provider is one module plus `@register`. Interface
Segregation — `LLMAdapter` covers completion only; embeddings have a different
shape and would get their own interface rather than widening this one. SDK
imports are lazy, inside each adapter's `__init__`, so an uninstalled provider
costs nothing until selected.

*The opaque-passthrough problem.* Normalising to neutral types and rebuilding
provider turns is lossy. Gemini's thinking models reject a replayed function
call whose `thought_signature` is missing. The fix is `ToolCall.provider_state`
— an opaque token the adapter captures and echoes back, documented as carried
but never interpreted. Any cross-provider abstraction needs such an escape
hatch for state it must preserve without understanding.

*Verification status.* Only the Gemini adapter is exercised against a live API.
OpenAI and Anthropic are written to their documented shapes but unverified.

---

## 5. Failure modes

| Failure | Cause | Present mitigation |
|---|---|---|
| Answer absent from corpus | nothing to retrieve | prompt rule + explicit refusal path |
| No relevance floor | `search()` always returns `top_k`, no `score_threshold` | **none — open gap** |
| Vocabulary mismatch | dense-only search is weak on identifiers and error codes | **none — hybrid search would fix it** |
| Wrong inferred filter | agentic path hides the right answer | filters printed; model retries on zero hits |
| Stale sources | deleted documents keep their vectors | **none — see G1** |
| Fragmented context | character chunking splits tables and procedures | heading carried in payload and prompt |
| Empty filtered result | over-narrow filter | `rag.answer()` refuses without calling the LLM |

---

## 6. Testing

`tests/test_adapter.py` runs with plain asserts, no pytest:

- all three providers register
- `SEARCH_DOCS` is plain JSON Schema and contains no `tenant_id`
- the full agent loop runs against a `FakeAdapter` — no network, no API key,
  no provider SDK
- the retry-after-empty-result behaviour

That the loop is testable without any SDK is the concrete payoff of D9.

Not covered: retrieval quality. There is no golden set and no recall@k
measurement — the largest testing gap, and the one that matters most.

---

## 7. Known gaps

| # | Gap | Notes |
|---|---|---|
| G1 | **No delete path.** `ingest.py` only upserts. Remove a file from `docs/` and its vectors remain, still retrieved and cited. | Fix: reconcile `doc_id`s present in the collection against those just ingested, delete the difference. ~15 lines. |
| G2 | **No relevance floor.** Every search returns `top_k` regardless of score. | Fix: `score_threshold`, and refuse when nothing clears it. |
| G3 | **No evaluation harness.** No golden set, no recall@k. | Highest-value addition: `COLLECTION_NAME` is env-driven, so competing indexes can be built side by side. |
| G4 | **Dense-only retrieval.** No BM25, no reranking. | Hybrid search plus a cross-encoder reranker is the standard next step. |
| G5 | **`GENERATION_MODEL` default unverified.** `gemini-3.5-flash` is the default in config, `.env.example` and the README. | Confirm against the current model list. |
| G6 | **Embeddings are Gemini-only.** `embeddings.py` calls the SDK directly. | Same adapter treatment, as a separate `EmbeddingAdapter`. |
| G7 | **No observability or budgets** in the agentic loop. | Tracing per tool call, timeouts, per-request cost ceiling. |

---

## 8. What would change at scale

Most decisions here are right for 23 chunks and wrong for 10M:

- D1 character chunking → structure-aware, with parent-document retrieval
- D2 payload-only metadata → still correct, and more valuable
- D6 full re-ingest → blue/green collections, `content_hash` to skip unchanged
- D8 agentic-by-default → router, with the fixed path for simple queries
- Deletion becomes mandatory, not optional (G1)
- HNSW actually engages: below Qdrant's indexing threshold (20k vectors by
  default) this collection is brute-forced and retrieval is exact, so recall is
  currently 100% and not yet a tunable
