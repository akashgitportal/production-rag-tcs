"""Load chunks + embeddings into Weaviate. Run from project root."""
import json
import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType

COLLECTION = "TCSDocs"


def main():
    with open("chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    vectors = np.load("embeddings.npy")
    assert len(chunks) == vectors.shape[0], "chunk/vector count mismatch!"

    client = weaviate.connect_to_local()
    try:
        if client.collections.exists(COLLECTION):
            client.collections.delete(COLLECTION)
            print(f"Deleted existing '{COLLECTION}'")

        client.collections.create(
            name=COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content",    data_type=DataType.TEXT),
                Property(name="source",     data_type=DataType.TEXT),
                Property(name="doc_type",   data_type=DataType.TEXT),
                Property(name="chunk_type", data_type=DataType.TEXT),
                Property(name="page",       data_type=DataType.INT),
                Property(name="sheet",      data_type=DataType.TEXT),
            ],
        )
        print(f"Created collection '{COLLECTION}'")

        collection = client.collections.get(COLLECTION)
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                m = chunk["metadata"]
                batch.add_object(
                    properties={
                        "content":    chunk["content"],
                        "source":     m.get("source", ""),
                        "doc_type":   m.get("doc_type", ""),
                        "chunk_type": m.get("chunk_type", ""),
                        "page":       m.get("page"),
                        "sheet":      m.get("sheet", ""),
                    },
                    vector=vector.tolist(),
                )
        failed = collection.batch.failed_objects
        if failed:
            print(f"WARNING: {len(failed)} failed. First: {failed[0]}")
        else:
            print("All objects imported with no failures.")
        total = collection.aggregate.over_all(total_count=True).total_count
        print(f"Objects now in '{COLLECTION}': {total}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
