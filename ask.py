"""CLI entry point. Usage:
    python ask.py "What was TCS's Q4 FY26 revenue?"
    python ask.py            # then type your question
"""
import sys
import time
from rag import RAGPipeline


def main():
    query = " ".join(sys.argv[1:]).strip() or input("Ask about TCS: ").strip()
    with RAGPipeline() as rag:
        start = time.time()
        result = rag.ask(query)
        elapsed = time.time() - start

        print("\nAnswer:", result["answer"])
        if result.get("blocked"):
            print(f"[blocked at {result['stage']}]")
        if "model" in result:
            fb = "  (fell back)" if result.get("fell_back") else ""
            print(f"[model: {result['model']}  |  tier: {result['tier']}{fb}]")
        if result.get("cache_hit"):
            print(f"[CACHE HIT  |  similarity: {result.get('cache_similarity')}]")
        print(f"[time: {elapsed:.2f}s]")
        if result.get("sources"):
            print("\nSources:")
            for s in result["sources"]:
                print("  -", s)


if __name__ == "__main__":
    main()