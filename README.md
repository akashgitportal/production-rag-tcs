# Production-Grade Multi-Model RAG (TCS Financial Documents)

A retrieval-augmented Q&A system over TCS investor-relations documents
(annual report PDF, fact-sheet PDF, financial-data XLSX), built with
production concerns first: guardrails, evaluation, and observability of quality.

## Architecture (query path)

    input guardrail  ->  retrieve (hybrid + rerank)  ->  generate  ->  output guardrail
      (rules + scope)      (Weaviate + Cohere)          (GPT)         (groundedness)

- **Ingestion** decomposes multi-format docs into metadata-tagged chunks, with
  per-row records for financial tables (fixes line-item recall).
- **Retrieval**: OpenAI embeddings in Weaviate, hybrid (vector + BM25) search,
  Cohere cross-encoder rerank. Measured best on a hand-built golden set:
  Hit@3 = 0.94, Hit@5 = 1.00, MRR = 0.82.
- **Guardrails**: cheap input checks (injection, scope) reject before spend;
  output groundedness check blocks unsupported answers.
- **Evaluation**: golden set + Hit@k / MRR harness to make changes data-driven.

## Setup

    pip install -r requirements.txt
    cp .env.example .env          # add your API keys
    docker compose up -d          # start Weaviate

## Build the index (one-time)

    python scripts/ingest.py      # docs -> chunks.jsonl
    python scripts/embed.py       # chunks.jsonl -> embeddings.npy
    python scripts/load.py        # -> Weaviate

## Ask a question

    python ask.py "What was TCS's Q4 FY26 revenue?"

## Evaluate retrieval

    python scripts/evaluate.py    # Hit@k / MRR across vector / hybrid / rerank
