# TCS Financial RAG — Production-Grade Multi-Model RAG System

A retrieval-augmented Q&A system over TCS investor-relations documents (annual report PDF, quarterly fact-sheet PDF, financial data-sheet XLSX), built with production concerns first: measured retrieval quality, input/output guardrails, cost-aware LLM routing, semantic caching, and a user + analytics UI.

Every number below was measured on this project's own golden-set evaluation harness — not copied from a tutorial.

## Measured results

| Retrieval config | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| vector only | 0.44 | 0.72 | 0.83 | 0.602 |
| hybrid (vector + BM25) | 0.33 | 0.83 | 0.89 | 0.567 |
| **hybrid + rerank** | **0.72** | **0.94** | **1.00** | **0.817** |

- **Semantic caching**: ~9.3s → ~0.5s on repeat questions once the server is warm (~17x faster), by skipping retrieval, reranking, generation, and the groundedness guardrail entirely for near-duplicate queries.
- **Multi-LLM routing**: simple lookups route to `gpt-4o-mini`; comparison/synthesis questions route to `gpt-4o`; automatic fallback if the primary model call fails.
- Evaluated on a 20-question golden set spanning easy single-fact lookups, cross-document synthesis, genuinely ambiguous figures (standalone vs consolidated segment revenue), and deliberately unanswerable questions.

## Architecture

Query flow, in order:

1. **Input guardrail** — free rule checks (length, prompt injection) plus an embedding-based domain-scope check. Rejects bad/off-topic queries before any expensive call.
2. **Semantic cache check** — Redis, cosine similarity ≥ 0.95 against past queries. On a hit, skip everything below.
3. **Query rewriter** — expands casual or ambiguous phrasing (e.g. "tcs 2026 revenue") into a retrieval-friendly form covering likely interpretations.
4. **Hybrid retrieval** — Weaviate, OpenAI embeddings + BM25 keyword search combined.
5. **Rerank** — Cohere cross-encoder re-orders the candidate pool for true relevance.
6. **Routed generation** — a rule-based classifier sends simple lookups to `gpt-4o-mini` and synthesis/comparison questions to `gpt-4o`, with automatic fallback on failure.
7. **Output guardrail** — an LLM judge verifies every claim in the answer is actually supported by the retrieved context; blocks unsupported or misattributed figures.
8. Cache the result, return the cited answer.

## What's implemented

- **Multi-format ingestion** — PDF (narrative + financial tables via `pymupdf4llm`) and XLSX (wide time-series reshaped into per-metric records). Financial tables are decomposed into per-row records in addition to whole-table chunks, fixing a measured recall gap on specific line items (e.g. "dividends paid") that were diluted inside large table chunks.
- **Hybrid retrieval + reranking** — OpenAI `text-embedding-3-large` vectors in Weaviate (hybrid vector + BM25 search), Cohere `rerank-v3.5` cross-encoder re-ordering the candidate pool.
- **Evaluation harness** — a 20-question golden set with content-marker-based scoring (Hit@1/3/5, MRR) across three retrieval configs. Used to catch and fix a bad ground-truth assumption during development (a question's "correct" answer turned out to have two legitimately different values — standalone vs consolidated segment revenue — which the golden set didn't originally account for).
- **Input guardrails** — free rule-based checks (length, prompt-injection patterns) before any paid API call, plus an embedding-based domain-scope check that rejects off-topic questions cheaply.
- **Output guardrails** — an LLM-based groundedness judge that verifies every claim in a generated answer is actually supported by the retrieved context, blocking answers with unsupported or misattributed figures.
- **Query rewriting** — a lightweight rewrite step expands casual or ambiguous phrasing into a retrieval-friendly query covering likely interpretations (fiscal year vs quarter), before embedding/search.
- **Multi-LLM routing** — a rule-based complexity classifier (keywords, query length) routes simple lookups to a cheap model and synthesis/comparison questions to a stronger model, with automatic fallback on failure.
- **Semantic caching** — Redis-backed cache keyed on query-embedding cosine similarity, skipping the full pipeline for near-duplicate questions.
- **Two UIs** — a plain end-user chat interface, and a creator-facing analytics dashboard (live request stats, cache hit rate, latency breakdown, routing tier split, and the retrieval evaluation table).

## Project structure

```
production-rag/
├── data/                 # TCS source documents (not included — see note below)
├── rag/
│   ├── guardrails.py     # input + output guardrails
│   ├── query_rewriter.py # casual-phrasing -> retrieval-friendly rewrite
│   ├── retriever.py      # embed -> hybrid search -> rerank
│   ├── router.py         # LLM tier routing + fallback
│   ├── generator.py      # cited, grounded answer generation
│   ├── cache.py          # Redis semantic cache
│   └── pipeline.py       # RAGPipeline: wires the full guarded flow
├── scripts/
│   ├── ingest.py         # documents -> chunks.jsonl
│   ├── embed.py          # chunks.jsonl -> embeddings.npy
│   ├── load.py           # embeddings.npy -> Weaviate
│   └── evaluate.py       # golden-set Hit@k / MRR evaluation harness
├── static/
│   ├── chat.html         # end-user chat UI
│   └── dashboard.html    # analytics dashboard UI
├── ask.py                # CLI entry point
├── app.py                # FastAPI server (chat + dashboard + /ask API)
├── docker-compose.yml    # Weaviate + Redis
├── golden_set.json       # 20-question retrieval evaluation set
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY and COHERE_API_KEY
docker compose up -d          # starts Weaviate + Redis
```

**Documents**: place your own TCS (or other) financial PDF/XLSX documents in `data/`. Source documents are not included in this repo (see Licensing note).

## Build the index (one-time)

```bash
python scripts/ingest.py      # documents -> chunks.jsonl
python scripts/embed.py       # chunks.jsonl -> embeddings.npy
python scripts/load.py        # -> Weaviate
python scripts/evaluate.py    # golden-set Hit@k / MRR (also saves eval_results.json)
```

## Run

CLI:

```bash
python ask.py "What was TCS's Q4 FY26 revenue?"
```

API + UIs:

```bash
uvicorn app:app --reload --port 8000
```

- Chat UI: `http://localhost:8000/ui/chat.html`
- Analytics dashboard: `http://localhost:8000/ui/dashboard.html`
- API docs: `http://localhost:8000/docs`

## Licensing note

This repo does not include the source TCS financial documents (annual report, fact sheet, data sheet). TCS's publicly filed investor-relations documents are copyrighted by Tata Consultancy Services; they are publicly available at tcs.com/investor-relations for anyone who wants to reproduce this project against the same corpus.