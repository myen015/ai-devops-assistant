import uuid
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

COLLECTION_NAME = "tech_docs"
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
client = QdrantClient(path="./qdrant_data")


def get_embedding(text: str) -> list[float]:
    return list(embed_model.embed([text]))[0].tolist()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Мәтінді қабаттасатын (overlap) бөліктерге бөледі."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def ingest_document(text: str, source_name: str, page: int = 1) -> int:
    """Құжатты бөліп, векторға айналдырып, метадеректерімен Qdrant-қа сақтайды."""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    first_vec = get_embedding(chunks[0])
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=len(first_vec), distance=Distance.COSINE),
        )

    points = []
    for idx, chunk in enumerate(chunks):
        vec = first_vec if idx == 0 else get_embedding(chunk)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": chunk,
                    "source": source_name,
                    "page": page,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search_docs(query: str, limit: int = 3) -> list[dict]:
    """Таза векторлық іздеу (Dense search)."""
    if not client.collection_exists(COLLECTION_NAME):
        return []
    query_vec = get_embedding(query)
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=limit,
    ).points
    return [
        {
            "id": str(h.id),
            "text": h.payload["text"],
            "source": h.payload["source"],
            "page": h.payload["page"],
            "score": round(h.score, 4),
        }
        for h in hits
    ]


def hybrid_search(query: str, limit: int = 3) -> list[dict]:
    """Dense Vector + BM25 Keyword Search (Reciprocal Rank Fusion арқылы біріктіру)."""
    if not client.collection_exists(COLLECTION_NAME):
        return []

    # 1. Барлық сақталған құжаттарды алып, BM25 (кілт сөз) бойынша іздеу
    all_points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1000)
    if not all_points:
        return []

    docs_map = {str(p.id): p.payload for p in all_points}
    doc_ids = list(docs_map.keys())
    tokenized_corpus = [docs_map[doc_id]["text"].lower().split() for doc_id in doc_ids]

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranked = sorted(zip(doc_ids, bm25_scores), key=lambda x: x[1], reverse=True)

    # 2. Векторлық (Dense) іздеу
    dense_hits = search_docs(query, limit=limit * 2)

    # 3. Reciprocal Rank Fusion (RRF) — екі рейтингті біріктіру
    rrf_scores: dict[str, float] = {}
    k = 60
    for rank, hit in enumerate(dense_hits):
        doc_id = hit["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, (doc_id, score) in enumerate(bm25_ranked[: limit * 2]):
        if score > 0:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:limit]
    return [
        {
            "text": docs_map[doc_id]["text"],
            "source": docs_map[doc_id]["source"],
            "page": docs_map[doc_id]["page"],
            "rrf_score": round(rrf_scores[doc_id], 4),
        }
        for doc_id in sorted_ids
    ]


def list_sources() -> list[dict]:
    """Базада индекстелген барлық бірегей файлдар мен олардың бөліктер санын қайтарады."""
    if not client.collection_exists(COLLECTION_NAME):
        return []
    all_points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1000)
    summary: dict[str, dict] = {}
    for p in all_points:
        src = p.payload["source"]
        if src not in summary:
            summary[src] = {"source": src, "chunks": 0, "pages": set()}
        summary[src]["chunks"] += 1
        summary[src]["pages"].add(p.payload["page"])

    return [
        {"source": v["source"], "chunks": v["chunks"], "pages": sorted(v["pages"])}
        for v in summary.values()
    ]


def delete_source(source_name: str) -> bool:
    """Көрсетілген файлдың барлық бөліктерін Qdrant-тан өшіреді."""
    if not client.collection_exists(COLLECTION_NAME):
        return False
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_name))]
            )
        ),
    )
    return True