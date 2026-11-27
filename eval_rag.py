import litellm
from pydantic import BaseModel, Field
from rag_qa import ask_docs
from rag_store import hybrid_search
from reranker import rerank_docs

# Эталондық тест жинағы (Golden Dataset)
GOLDEN_DATASET = [
    {
        "question": "What does ERR_504 mean?",
        "expected_found": True,
        "expected_source": "errors.md",
        "expected_fact": "Upstream gateway timed out after 30 seconds",
    },
    {
        "question": "What is the API rate limit?",
        "expected_found": True,
        "expected_source": "api_spec.md",
        "expected_fact": "100 requests per minute",
    },
    {
        "question": "What tool is used for GitOps deployment?",
        "expected_found": True,
        "expected_source": "infra_guide.pdf",
        "expected_fact": "ArgoCD is used for GitOps deployment",
    },
    {
        "question": "How do we deploy to Mars colonies?",
        "expected_found": False,
        "expected_source": None,
        "expected_fact": "Not found in provided documents",
    },
]


class JudgeVerdict(BaseModel):
    faithfulness_score: int = Field(..., ge=1, le=5, description="5 if completely accurate and aligned, 1 if wrong or hallucinated")
    passed: bool = Field(..., description="True if actual_answer contains the expected_fact without contradicting it")
    reason: str


def judge_answer(question: str, expected_fact: str, actual_answer: str, model: str = "ollama/qwen2.5:3b") -> JudgeVerdict:
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI QA evaluator. Check if actual_answer includes the core meaning of expected_fact. "
                    "Rules:\n"
                    "1. Extra true context in actual_answer is ALLOWED and should NOT fail the test.\n"
                    "2. Only set passed=false if expected_fact is missing or contradicted.\n"
                    "3. Use faithfulness_score=5 for a pass, and 1 for a fail."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Expected Fact: {expected_fact}\n"
                    f"Actual Answer: {actual_answer}"
                ),
            },
        ],
        response_format=JudgeVerdict,
        temperature=0.0,
    )
    return JudgeVerdict.model_validate_json(response.choices[0].message.content)


def run_evals():
    passed_count = 0
    retrieval_hits = 0
    positive_cases = [c for c in GOLDEN_DATASET if c["expected_source"] is not None]
    total = len(GOLDEN_DATASET)

    for idx, case in enumerate(GOLDEN_DATASET, 1):
        # 1. Retrieval Recall@3 тексерісі (іздеу кезеңінің таза дәлдігі)
        if case["expected_source"]:
            candidates = hybrid_search(case["question"], limit=6)
            top_chunks = rerank_docs(case["question"], candidates, top_k=3)
            if any(c["source"] == case["expected_source"] for c in top_chunks):
                retrieval_hits += 1

        # 2. Толық RAG (Retrieval + Generation) тексерісі
        rag_res = ask_docs(case["question"])
        source_ok = (
            rag_res.found_in_docs == case["expected_found"]
            and (
                case["expected_source"] is None
                or any(c.source == case["expected_source"] for c in rag_res.citations)
            )
        )

        verdict = judge_answer(case["question"], case["expected_fact"], rag_res.answer)
        case_passed = source_ok and verdict.passed
        if case_passed:
            passed_count += 1

        status = "PASS" if case_passed else "FAIL"
        print(f"[{status}] Case {idx}: {case['question']}")
        print(f"   Answer: {rag_res.answer} | Citations: {[c.model_dump() for c in rag_res.citations]}")
        print(f"   Judge: {verdict.faithfulness_score}/5 — {verdict.reason}\n")

    recall_at_3 = round((retrieval_hits / len(positive_cases)) * 100, 1)
    end_to_end_acc = round((passed_count / total) * 100, 1)
    print(f"=== RETRIEVAL RECALL@3: {retrieval_hits}/{len(positive_cases)} ({recall_at_3}%) ===")
    print(f"=== END-TO-END EVAL SUMMARY: {passed_count}/{total} passed ({end_to_end_acc}%) ===")


if __name__ == "__main__":
    run_evals()