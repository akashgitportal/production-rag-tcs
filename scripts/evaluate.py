"""
Step 14 — Retrieval evaluation harness.
Scores three configs (vector / hybrid / hybrid+rerank) on golden_set.json
using Hit@k and MRR. This is how we decide what actually helps, with numbers.
"""
import os
import json
from pathlib import Path
import weaviate
from weaviate.classes.query import MetadataQuery
from openai import OpenAI
import cohere
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

COLLECTION = "TCSDocs"
EMBED_MODEL = "text-embedding-3-large"
RERANK_MODEL = "rerank-v3.5"
K = 5                      # we retrieve/score top-5
CANDIDATES = 30           # match production retriever depth
ROOT = Path(__file__).resolve().parents[1]
GOLDEN_CANDIDATES = [
    ROOT / "golden_set.json",
    ROOT / "data" / "golden_set.json",
    ROOT / "scripts" / "golden_set.json",
]


def load_golden_set():
    for path in GOLDEN_CANDIDATES:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

    sample = [
        {
            "id": 1,
            "question": "What was TCS's Q4 FY26 revenue?",
            "markers": ["revenue", "fy26", "q4"],
            "difficulty": "answerable",
        },
        {
            "id": 2,
            "question": "How much was TCS's free cash flow?",
            "markers": ["free cash flow", "cash flow"],
            "difficulty": "answerable",
        },
        {
            "id": 3,
            "question": "What is the capital expenditure guidance for FY26?",
            "markers": ["capital expenditure", "capex", "guidance"],
            "difficulty": "answerable",
        },
    ]
    target = ROOT / "golden_set.json"
    with target.open("w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"Created starter golden set at {target}")
    return sample


def embed_query(text):
    return openai_client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


# ---- three retrieval strategies, each returns a list of content strings ---- #
def retrieve_vector(col, query, qvec):
    objs = col.query.near_vector(near_vector=qvec, limit=K).objects
    return [o.properties["content"] for o in objs]


def retrieve_hybrid(col, query, qvec):
    objs = col.query.hybrid(query=query, vector=qvec, alpha=0.5, limit=K).objects
    return [o.properties["content"] for o in objs]


def retrieve_rerank(col, query, qvec):
    objs = col.query.hybrid(query=query, vector=qvec, alpha=0.5,
                            limit=CANDIDATES).objects
    docs = [o.properties["content"] for o in objs]
    if not docs:
        return []
    res = co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=K)
    return [docs[r.index] for r in res.results]


# ---- scoring ---- #
def first_hit_rank(contents, markers):
    """Return 1-based rank of the first chunk containing any marker, else None."""
    for i, content in enumerate(contents):
        low = content.lower()
        if any(m.lower() in low for m in markers):
            return i + 1
    return None


def evaluate(col, golden):
    configs = {
        "vector":         retrieve_vector,
        "hybrid":         retrieve_hybrid,
        "hybrid+rerank":  retrieve_rerank,
    }
    # accumulate metrics per config
    scores = {name: {"hit@1": 0, "hit@3": 0, "hit@5": 0, "rr": 0.0}
              for name in configs}

    answerable = [q for q in golden if q["difficulty"] != "unanswerable"]
    unanswerable = [q for q in golden if q["difficulty"] == "unanswerable"]

    per_question = []  # for detailed printout

    for q in answerable:
        qvec = embed_query(q["question"])
        row = {"id": q["id"], "q": q["question"][:45]}
        for name, fn in configs.items():
            contents = fn(col, q["question"], qvec)
            rank = first_hit_rank(contents, q["markers"])
            if rank:
                if rank == 1:
                    scores[name]["hit@1"] += 1
                if rank <= 3:
                    scores[name]["hit@3"] += 1
                if rank <= 5:
                    scores[name]["hit@5"] += 1
                scores[name]["rr"] += 1.0 / rank
            row[name] = rank if rank else "-"
        per_question.append(row)

    n = len(answerable)
    return scores, n, per_question, unanswerable


def main():
    golden = load_golden_set()
    client = weaviate.connect_to_local()
    try:
        col = client.collections.get(COLLECTION)
        scores, n, per_question, unanswerable = evaluate(col, golden)

        # ---- per-question ranks (rank of first correct chunk; '-' = miss) ----
        print("\nPer-question rank of first correct chunk ('-' = not in top 5):")
        print(f"{'id':<5}{'question':<47}{'vector':<9}{'hybrid':<9}{'rerank':<9}")
        print("-" * 79)
        for r in per_question:
            print(f"{r['id']:<5}{r['q']:<47}{str(r['vector']):<9}"
                  f"{str(r['hybrid']):<9}{str(r['hybrid+rerank']):<9}")

        # ---- summary metrics ----
        print("\n" + "=" * 60)
        print(f"RETRIEVAL METRICS  (n={n} answerable questions)")
        print("=" * 60)
        print(f"{'config':<16}{'Hit@1':<9}{'Hit@3':<9}{'Hit@5':<9}{'MRR':<8}")
        print("-" * 60)
        for name, s in scores.items():
            print(f"{name:<16}"
                  f"{s['hit@1']/n:<9.2f}{s['hit@3']/n:<9.2f}"
                  f"{s['hit@5']/n:<9.2f}{s['rr']/n:<8.3f}")
        print("=" * 60)
        # ---- save results for the dashboard (analytics/eval endpoint) ----
        results_out = {
            "n_questions": n,
            "metrics": {
                name: {
                    "hit@1": round(s["hit@1"] / n, 3),
                    "hit@3": round(s["hit@3"] / n, 3),
                    "hit@5": round(s["hit@5"] / n, 3),
                    "mrr": round(s["rr"] / n, 3),
                }
                for name, s in scores.items()
            },
        }
        out_path = ROOT / "eval_results.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results_out, f, ensure_ascii=False, indent=2)
        print(f"Saved {out_path} (used by the dashboard UI)")

        # ---- unanswerable: just show what hybrid retrieves (eyeball) ----
        print("\nUnanswerable questions (retrieval still returns something; the")
        print("generator must learn to refuse these later):")
        for q in unanswerable:
            qvec = embed_query(q["question"])
            top = retrieve_hybrid(col, q["question"], qvec)
            print(f"  [{q['id']}] {q['question']}")
            print(f"        top chunk: {top[0][:80] if top else '(none)'}...")

    finally:
        client.close()


if __name__ == "__main__":
    main()
