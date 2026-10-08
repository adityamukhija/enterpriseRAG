# Enterprise RAG Assistant

A Retrieval-Augmented Generation service built with FastAPI, combining dense
vector search and keyword search over documents, with cost-aware LLM routing
and semantic response caching.

## How it works

**Ingestion** (`POST /ingest`): a document is loaded, split into overlapping
chunks (recursive character splitting), embedded with Gemini, and stored in
Postgres via `pgvector`. A BM25 index is rebuilt over the full corpus.

**Query** (`POST /query`):
1. Check the semantic cache (Redis) for a previously answered question with
   a similar embedding (cosine similarity above a threshold) — skips the LLM
   entirely on a hit.
2. Retrieve candidates from both vector search (`pgvector`, HNSW index,
   cosine distance) and BM25 keyword search, merged with Reciprocal Rank
   Fusion (RRF).
3. Classify query complexity with a cheap model call, then route to a
   budget model (`gemini-2.0-flash`) for simple questions or a flagship
   model (`gemini-2.0-pro`) for ones needing reasoning/synthesis.
4. Generate the answer grounded in the retrieved chunks, return it with
   source citations, and cache it for next time.

All endpoints except `/health` and `/auth/token` require a JWT bearer token.

## Stack

- **API**: FastAPI
- **Vector store**: PostgreSQL + `pgvector` (HNSW index, cosine distance)
- **Keyword search**: BM25 (`rank-bm25`)
- **Cache**: Redis, embedding-similarity based
- **Embeddings & generation**: Google Gemini
- **Auth**: JWT (`python-jose`)

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector)
  extension available
- Redis
- A [Gemini API key](https://ai.google.dev/)

### Install

```bash
python -m venv rag-env
source rag-env/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp app/.env.example app/.env
# edit app/.env and set GEMINI_API_KEY
```

Other settings (database URL, Redis URL, chunk size, model names, JWT
secret, etc.) have defaults in `app/config.py` and can be overridden via
environment variables of the same name.

### Run

```bash
uvicorn app.main:app --reload
```

The API docs are available at `http://localhost:8000/docs`.

## Usage

Get a token:

```bash
curl -X POST "http://localhost:8000/auth/token?user_id=me"
```

Ingest a document:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer <token>" \
  -F "file=@documents/sample.txt"
```

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the cancellation policy?"}'
```

## Project layout

```
app/
  main.py                 FastAPI app and routes
  config.py               Settings (env-driven)
  auth.py                 JWT issuing/verification
  database.py             SQLAlchemy engine/session, pgvector setup
  models.py               ORM + request/response models
  ingestion/
    loader.py             PDF/text loading
    chunker.py            Recursive chunking with overlap
    embedder.py           Gemini embeddings
  retrieval/
    vector_store.py       pgvector similarity search
    bm25.py                BM25 keyword search
    hybrid.py              RRF fusion of both
  generation/
    llm.py                 Answer generation
    router.py               Complexity-based model routing
    prompts.py               Versioned prompt templates
  cache/
    semantic_cache.py       Redis-backed semantic cache
```
