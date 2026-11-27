import litellm
from pydantic import BaseModel, Field
from rag_store import hybrid_search
from reranker import rerank_docs


class Citation(BaseModel):
    source: str = Field(..., description="Source file name")
    page: int = Field(..., description="Page number")


class RagAnswer(BaseModel):
    answer: str = Field(..., description="Answer strictly based on the provided context")
    citations: list[Citation] = Field(default_factory=list)
    found_in_docs: bool = Field(..., description="True if answer is supported by context, else False")


def ask_docs(question: str, model: str = "ollama/qwen2.5:3b", limit: int = 3) -> RagAnswer:
    candidates = hybrid_search(question, limit=limit * 2)
    chunks = rerank_docs(question, candidates, top_k=limit)

    if not chunks:
        return RagAnswer(
            answer="No documents indexed yet.",
            citations=[],
            found_in_docs=False,
        )

    context_blocks = [
        f"Source: {c['source']} | Page: {c['page']}\n{c['text']}"
        for c in chunks
    ]
    context_text = "\n\n---\n\n".join(context_blocks)

    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict technical documentation assistant. "
                    "Answer the question ONLY using the provided context. "
                    "Always include the source and page in 'citations' when found_in_docs is true. "
                    "If the context does not contain the answer, set found_in_docs=false, "
                    "citations=[], and say 'Not found in provided documents.' "
                    "Do not use outside knowledge."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            },
        ],
        response_format=RagAnswer,
        temperature=0.0,
    )
    result = RagAnswer.model_validate_json(response.choices[0].message.content)

    if result.found_in_docs and not result.citations:
        top = chunks[0]
        result.citations = [Citation(source=top["source"], page=top["page"])]
    elif not result.found_in_docs:
        result.citations = []

    return result