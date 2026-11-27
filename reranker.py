from fastembed.rerank.cross_encoder import TextCrossEncoder

# MacBook Air үшін өте жеңіл және жылдам Cross-Encoder (~90 MB)
reranker_model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


def rerank_docs(query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
    """Табылған құжаттарды сұраққа нақты сәйкестігі бойынша қайта сұрыптайды."""
    if not docs:
        return []

    texts = [d["text"] for d in docs]
    scores = list(reranker_model.rerank(query, texts))

    scored_docs = []
    for doc, score in zip(docs, scores):
        updated = dict(doc)
        updated["rerank_score"] = round(float(score), 4)
        scored_docs.append(updated)

    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_docs[:top_k]