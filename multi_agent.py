import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from pydantic import BaseModel, Field
from analyzer import analyze_logs
from rag_qa import ask_docs


class TriageResult(BaseModel):
    category: str
    priority: str
    search_query: str
    has_raw_logs: bool


class IncidentReport(BaseModel):
    priority: str
    category: str
    root_cause: str
    action_steps: list[str]
    citations: list[dict] = Field(default_factory=list)


def triage_agent(user_input: str) -> TriageResult:
    """1-агент (Fast Router): Сұранысты 0.001 сек ішінде жіктейді (3B модель циклге түспеуі үшін)."""
    text_upper = user_input.upper()
    has_logs = any(tag in text_upper for tag in ("ERROR", "WARN", "EXCEPTION", "TRACEBACK", "[202"))
    has_question = "?" in user_input or any(
        w in user_input.lower() for w in ("what", "how", "why", "explain", "mean", "docs")
    )

    if has_logs and has_question:
        category = "hybrid"
    elif has_logs:
        category = "log_only"
    else:
        category = "doc_only"

    priority = "P1" if ("ERROR" in text_upper or "ERR_" in text_upper) else ("P2" if "WARN" in text_upper else "P3")

    # Қысқа іздеу сұрағын бөліп алу
    query_part = user_input.split("]")[-1].strip() if "]" in user_input else user_input
    return TriageResult(
        category=category,
        priority=priority,
        search_query=query_part[:120],
        has_raw_logs=has_logs,
    )


def researcher_agent(user_input: str, triage: TriageResult) -> dict:
    """2-агент: Triage нұсқауы бойынша нақты деректерді жинайды."""
    evidence: dict = {"log_analysis": None, "doc_answer": None, "citations": []}

    if triage.has_raw_logs or triage.category in ("log_only", "hybrid"):
        log_report = analyze_logs(user_input)
        evidence["log_analysis"] = log_report.model_dump()

    if triage.category in ("doc_only", "hybrid"):
        rag_res = ask_docs(triage.search_query)
        evidence["doc_answer"] = rag_res.answer
        evidence["citations"] = [c.model_dump() for c in rag_res.citations]

    return evidence


def responder_agent(
    user_input: str,
    triage: TriageResult,
    evidence: dict,
    model: str = "ollama/qwen2.5:3b",
) -> IncidentReport:
    """3-агент: Тек жиналған фактілер негізінде қорытынды есеп пен шешім жазады."""
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Senior Incident Responder. Based ONLY on the evidence below, "
                    "write a short root_cause (1 sentence) and 2 concrete action_steps."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Request: {user_input}\n"
                    f"Logs: {evidence['log_analysis']}\n"
                    f"Docs: {evidence['doc_answer']}"
                ),
            },
        ],
        response_format=IncidentReport,
        temperature=0.0,
        repeat_penalty=1.15,
        max_tokens=400,
        timeout=120,
    )
    report = IncidentReport.model_validate_json(response.choices[0].message.content)
    report.priority = triage.priority
    report.category = triage.category
    report.citations = evidence["citations"]
    return report


def run_incident_pipeline(user_input: str, model: str = "ollama/qwen2.5:3b") -> IncidentReport:
    """Толық Multi-Agent тізбегі: Triage -> Researcher -> Responder."""
    triage = triage_agent(user_input)
    evidence = researcher_agent(user_input, triage)
    return responder_agent(user_input, triage, evidence, model=model)