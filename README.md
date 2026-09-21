# RAG Engine from Scratch

### Retrieval-Augmented Generation on Gemini + Qdrant, built without LangChain

A complete RAG pipeline implemented in plain Python. Chunking, embedding,
vector storage, retrieval and grounded generation are each a small, readable
module in `src/` — no framework abstractions in between.

The engine indexes a corpus of engineering documentation (service runbooks,
architecture notes, incident reports) and answers natural-language questions
strictly from what it retrieves, citing the source file for every answer and
declining when the corpus does not contain one.

**Stack:** Python 3.11 · Gemini (`gemini-embedding-001`) · Qdrant · Docker

Design decisions, tradeoffs and known gaps are documented in [DESIGN.md](DESIGN.md).

## Architecture

```
docs/*.md, *.txt
      │
      ▼
 loader.py        read files → documents
      │
      ▼
 chunker.py       split into overlapping chunks (CHUNK_SIZE / CHUNK_OVERLAP)
      │
      ▼
 embeddings.py    Gemini embedding API  (gemini-embedding-001)
      │
      ▼
 vector_store.py  Qdrant (local Docker) — vector + chunk text + source metadata
      │
      │   ─────────────── ingest ends here (scripts/ingest.py) ───────────────
      │
      │            question ──► embeddings.py ──► Qdrant similarity search
      │                                                  │
      ▼                                                  ▼
 rag.py           top-K chunks ──► grounded prompt ──► Gemini LLM ──► answer
                                                    (gemini-3.5-flash)
```

## Project structure

```
rag-project/
├── README.md
├── DESIGN.md             # design decisions, tradeoffs, known gaps
├── requirements.txt
├── .env.example
├── docs/                 # sample engineering docs (Acme Commerce)
├── src/
│   ├── config.py         # all settings, read from env / .env
│   ├── loader.py         # read .md / .txt files + document metadata
│   ├── chunker.py        # sliding-window chunking + chunk metadata
│   ├── embeddings.py     # embed_text(), embed_documents()
│   ├── vector_store.py   # Qdrant create / index / upsert / search
│   ├── filters.py        # user parameters → Qdrant filter
│   ├── rag.py            # retrieve → build_prompt → generate
│   ├── agent.py          # agentic retrieval: model picks its own filters
│   ├── main.py           # interactive REPL
│   └── llm/              # provider-neutral LLM access (Adapter pattern)
├── scripts/
│   ├── ingest.py         # docs → Qdrant
│   └── ask.py            # question → answer, with explicit filters
└── tests/
    └── test_adapter.py   # agent loop driven by a FakeAdapter
```

## Prerequisites

| Tool | Version | Verify with |
|---|---|---|
| Python | 3.11+ | `python3.11 --version` |
| Docker | 20.10+ | `docker --version` |
| Gemini API key | – | create one at https://aistudio.google.com/apikey |

**Python 3.11+**

```bash
brew install python@3.11                          # macOS
sudo apt install python3.11 python3.11-venv       # Ubuntu / Debian
```

Windows: use the installer from https://www.python.org/downloads/.

**Docker** — Qdrant runs as a local container, so Docker must be installed *and*
the daemon must be running before step 4 of Setup.

- macOS / Windows: install Docker Desktop from
  https://www.docker.com/products/docker-desktop and launch it.
- Linux: `curl -fsSL https://get.docker.com | sh`, then
  `sudo usermod -aG docker $USER` and log out and back in.

Check both the CLI and the daemon:

```bash
docker --version     # prints a version if the CLI is installed
docker info          # must succeed — this is the one that proves the daemon is up
```

If `docker info` fails with `Cannot connect to the Docker daemon`, Docker is
installed but not running: start Docker Desktop, or `sudo systemctl start docker`
on Linux, then re-run `docker info` before continuing.

## Setup

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Gemini API key
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...   (or: export GEMINI_API_KEY=...)

# 4. Start Qdrant locally
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
# Dashboard: http://localhost:6333/dashboard

# 5. Ingest the sample documents
python scripts/ingest.py

# 6. Ask questions
python scripts/ask.py "How is payment-service deployed?"
python -m src.main          # or: interactive loop (model picks its own filters)

# 7. Stop Qdrant when done
docker stop qdrant
docker rm qdrant            # optional; data persists in ./qdrant_storage
```

## Configuration

All values are read in `src/config.py` and can be set in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | required |
| `QDRANT_URL` | `http://localhost:6333` | where Qdrant listens |
| `COLLECTION_NAME` | `engineering_docs` | Qdrant collection |
| `TENANT_ID` | `default` | tenant stamped at ingest and enforced on every search |
| `TOP_K` | `4` | chunks retrieved per question |
| `CHUNK_SIZE` | `800` | characters per chunk (~200 tokens) |
| `CHUNK_OVERLAP` | `150` | characters shared between adjacent chunks |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | GA embedding model |
| `EMBEDDING_DIM` | `768` | output dims (768 / 1536 / 3072 supported) |
| `GENERATION_MODEL` | `gemini-3.5-flash` | LLM for the final answer |
| `LLM_PROVIDER` | `gemini` | `gemini` \| `openai` \| `anthropic` (see `src/llm/`) |

If you change `EMBEDDING_MODEL` or `EMBEDDING_DIM`, delete the collection first
(dashboard or `curl -X DELETE localhost:6333/collections/engineering_docs`) and re-ingest — vectors of different sizes cannot live in one collection.

## Metadata filtering

Each chunk carries metadata extracted from the documents themselves —
`tenant_id`, `doc_type`, `heading`, `services`, `severity`, `date`, `owner`,
plus lifecycle fields (`content_hash`, `ingested_at`, `batch_id`). None of it is
added to the embedded text, so filters can change without re-embedding.

```bash
python scripts/ask.py "What went wrong with payments?" --doc-type incident
python scripts/ask.py "How do we roll back?" --service payment-service
python scripts/ask.py "Root cause?" --date-from 2026-03-01 --date-to 2026-03-31
python scripts/ask.py "What are the rate limits?" --source api.md
python scripts/ask.py --help          # all filters
```

`--doc-type`, `--source`, `--doc-id`, `--service`, `--severity` and `--owner`
are repeatable and OR together; different filters AND together. An unknown
filter name is rejected rather than silently ignored. Filters are applied
*during* the vector search, so `TOP_K` chunks come from the matching subset.

Metadata lives in the Qdrant payload, not in the vector, so refreshing it needs
no embedding calls:

```bash
python scripts/ingest.py --payload-only
```

Valid only while chunk boundaries are unchanged — change `CHUNK_SIZE` or the
chunking logic and a full `python scripts/ingest.py` is required.

## Two ways to query

**Explicit filters — `scripts/ask.py`.** The caller states the filters. Use this
for scripting and for measuring retrieval, where filters must be held constant.

**Agentic retrieval — `python -m src.main`.** Retrieval is exposed to the model
as a `search_docs` tool, so it picks filters itself and can search again if one
returns nothing:

```
> have we had any SEV-1 incidents?

SEARCHES:
  [1] query='SEV-1'   filters: tenant=default, doc_type=incident, severity=SEV-1
      hits: 0
  [2] query='SEV-1'   filters: tenant=default
      hits: 4

ANSWER:
I could not find this in the documentation.
```

Every search is printed, because a wrongly-inferred filter silently hides the
right answer. `python -m src.main --plain` bypasses tool calling for comparison.

`tenant_id` is never a tool parameter — it is injected server-side. See
[DESIGN.md](DESIGN.md) §D3.

## Swapping the LLM provider

`src/llm/` puts the three providers behind one interface, so `src/agent.py`
imports no provider SDK and the retrieval loop is written once:

```bash
LLM_PROVIDER=openai GENERATION_MODEL=gpt-4o python -m src.main
python tests/test_adapter.py     # drives the whole loop with a FakeAdapter
```

Only the Gemini adapter is verified against a live API. Design rationale, the
per-provider envelope differences and the opaque-passthrough problem are in
[DESIGN.md](DESIGN.md) §D9.

## Example questions

```bash
python scripts/ask.py "How is payment-service deployed?"
python scripts/ask.py "What caused incident 101?"
python scripts/ask.py "Which services are synchronous in the checkout flow and why?"
python scripts/ask.py "What is the rate limit on the public API?"
python scripts/ask.py "How do I roll back payment-service quickly?"
python scripts/ask.py "What should I do if orders are stuck in RESERVED?"
python scripts/ask.py "Which databases have point-in-time recovery enabled?"
python scripts/ask.py "How many replicas does order-service run in prod?"
python scripts/ask.py "When do new engineers join the on-call rotation?"
python scripts/ask.py "What is the company's vacation policy?"   # not in docs → should say so
```

## Data flow, step by step

1. **Load** – `loader.py` reads each `.md`/`.txt` file into a dict with an id, filename and text.
2. **Chunk** – `chunker.py` slides an 800-character window with 150 characters of overlap, preferring to cut at paragraph or sentence ends. Each chunk remembers its source file.
3. **Embed** – `embeddings.py` sends chunk texts to Gemini in batches of up to 100 with `task_type=RETRIEVAL_DOCUMENT` and gets back 768-dim vectors.
4. **Store** – `vector_store.py` upserts one Qdrant point per chunk: the vector plus a payload holding the chunk text and metadata. Point ids are deterministic UUIDs, so re-running ingest overwrites rather than duplicates.
5. **Retrieve** – At query time the question is embedded with `task_type=RETRIEVAL_QUERY` and Qdrant returns the `TOP_K` nearest chunks by cosine similarity.
6. **Prompt** – `rag.py` places the chunks, each labelled with its source, into a grounded prompt that forbids inventing facts and asks for cited filenames.
7. **Generate** – The prompt goes to `gemini-3.5-flash`; the answer and the retrieved chunks (with scores) are printed so you can judge retrieval quality.

## Things worth experimenting with

- Lower `TOP_K` to 1–2 and watch answers lose context; raise it to 8 and watch noise creep in.
- Set `CHUNK_SIZE=300` and re-ingest; note how retrieved chunks become more precise but lose surrounding context.
- Ask a question whose answer spans two files (e.g. incident 101 + troubleshooting) and check whether both sources are retrieved.
