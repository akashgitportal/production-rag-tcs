"""FastAPI endpoint wrapping the guarded RAG pipeline, plus two static UIs:
  /ui/chat.html       - end-user Q&A chat interface
  /ui/dashboard.html  - creator analytics dashboard (live stats + eval metrics)
"""
import time
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import RAGPipeline

_state: dict = {}

_stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "blocked_input": 0,
    "blocked_output": 0,
    "tier_simple": 0,
    "tier_complex": 0,
    "fell_back": 0,
    "latency_cache_hit": [],
    "latency_cache_miss": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: connecting to Weaviate, Redis, OpenAI, Cohere...")
    _state["rag"] = RAGPipeline()
    print("RAGPipeline ready.")
    yield
    print("Shutting down: closing connections...")
    _state["rag"].close()


app = FastAPI(title="TCS Financial RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    blocked: bool
    stage: str = None
    sources: list = []
    model: str = None
    tier: str = None
    fell_back: bool = False
    cache_hit: bool = False
    cache_similarity: float = None
    rewritten_query: str = None
    latency_seconds: float
    debug_draft: str = None


@app.get("/health")
def health():
    rag = _state["rag"]
    checks = {}
    try:
        checks["weaviate"] = rag.client.is_ready()
    except Exception as e:
        checks["weaviate"] = f"error: {e}"
    try:
        checks["redis"] = rag.cache.r.ping()
    except Exception as e:
        checks["redis"] = f"error: {e}"
    healthy = all(v is True for v in checks.values())
    return {"status": "healthy" if healthy else "degraded", "checks": checks}


@app.get("/cache/stats")
def cache_stats():
    return _state["rag"].cache.stats()


@app.post("/cache/clear")
def cache_clear():
    n = _state["rag"].cache.clear()
    return {"cleared_entries": n}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    rag = _state["rag"]
    start = time.time()
    result = rag.ask(req.question)
    elapsed = round(time.time() - start, 3)
    result["latency_seconds"] = elapsed

    _stats["total_requests"] += 1
    if result.get("stage") == "input_guardrail":
        _stats["blocked_input"] += 1
    if result.get("stage") == "output_guardrail":
        _stats["blocked_output"] += 1
    if result.get("cache_hit"):
        _stats["cache_hits"] += 1
        _stats["latency_cache_hit"].append(elapsed)
    else:
        _stats["latency_cache_miss"].append(elapsed)
    if result.get("tier") == "simple":
        _stats["tier_simple"] += 1
    elif result.get("tier") == "complex":
        _stats["tier_complex"] += 1
    if result.get("fell_back"):
        _stats["fell_back"] += 1

    return result


def _avg(lst):
    return round(sum(lst) / len(lst), 3) if lst else None


@app.get("/analytics/live")
def analytics_live():
    total = _stats["total_requests"]
    cache_hits = _stats["cache_hits"]
    return {
        "total_requests": total,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total, 3) if total else None,
        "blocked_input": _stats["blocked_input"],
        "blocked_output": _stats["blocked_output"],
        "tier_simple": _stats["tier_simple"],
        "tier_complex": _stats["tier_complex"],
        "fell_back": _stats["fell_back"],
        "avg_latency_cache_hit": _avg(_stats["latency_cache_hit"]),
        "avg_latency_cache_miss": _avg(_stats["latency_cache_miss"]),
    }


@app.get("/analytics/eval")
def analytics_eval():
    path = Path("eval_results.json")
    if not path.exists():
        return {"available": False,
                "message": "Run scripts/evaluate.py to generate eval_results.json"}
    return {"available": True, **json.loads(path.read_text(encoding="utf-8"))}


app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
