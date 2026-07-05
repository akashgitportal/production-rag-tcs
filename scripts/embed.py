"""Embed all chunks in chunks.jsonl -> embeddings.npy. Run from project root."""
import json
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

CHUNKS_FILE = "chunks.jsonl"
EMB_FILE = "embeddings.npy"
MODEL = "text-embedding-3-large"
BATCH_SIZE = 100


def embed_batch(texts, retries=3):
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(model=MODEL, input=texts)
            return [d.embedding for d in resp.data], resp.usage.total_tokens
        except Exception as e:
            wait = 2 ** attempt
            print(f"  batch failed ({e}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Batch failed after retries")


def main():
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    print(f"Loaded {len(chunks)} chunks. Embedding with {MODEL}...")

    all_vectors, total_tokens, start = [], 0, time.time()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectors, tokens = embed_batch([c["content"] for c in batch])
        all_vectors.extend(vectors)
        total_tokens += tokens
        print(f"  embedded {i + len(batch)}/{len(chunks)}  (tokens: {total_tokens})")

    matrix = np.array(all_vectors, dtype=np.float32)
    np.save(EMB_FILE, matrix)
    assert matrix.shape[0] == len(chunks), "vector/chunk count mismatch!"

    cost = total_tokens / 1_000_000 * 0.13
    print("\n" + "=" * 55)
    print(f"Embedded {matrix.shape[0]} chunks -> {EMB_FILE}")
    print(f"Vector shape : {matrix.shape}")
    print(f"Total tokens : {total_tokens:,}   Est. cost: ${cost:.4f}")
    print(f"Time         : {time.time() - start:.1f}s")
    print("=" * 55)


if __name__ == "__main__":
    main()
