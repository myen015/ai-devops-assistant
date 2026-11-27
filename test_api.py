from fastapi.testclient import TestClient
import api
from analyzer import LogIssue, LogReport
from rag_qa import Citation, RagAnswer

client = TestClient(api.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint(monkeypatch):
    def fake_analyze_logs(log_text: str, model: str = "ollama/qwen2.5:3b") -> LogReport:
        return LogReport(
            summary="Database connection error found.",
            issues=[
                LogIssue(
                    service="db-service",
                    severity="high",
                    reason="Connection refused on port 5432",
                )
            ],
        )

    monkeypatch.setattr(api, "analyze_logs", fake_analyze_logs)

    payload = {
        "log_text": "2026-11-23 12:00:01 ERROR db-service: Connection refused on port 5432"
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["issues"]) == 1
    assert data["issues"][0]["service"] == "db-service"


def test_ingest_endpoint(monkeypatch):
    def fake_ingest_document(text: str, source_name: str, page: int = 1) -> int:
        return 2

    monkeypatch.setattr(api, "ingest_document", fake_ingest_document)

    payload = {
        "text": "Service mesh retry policy is set to 3 attempts.",
        "source_name": "network.md",
        "page": 4,
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "status": "indexed",
        "chunks_added": 2,
        "source": "network.md",
    }


def test_query_endpoint(monkeypatch):
    def fake_ask_docs(question: str, model: str = "ollama/qwen2.5:3b") -> RagAnswer:
        return RagAnswer(
            answer="Error code ERR_504 means upstream gateway timeout.",
            citations=[Citation(source="errors.md", page=5)],
            found_in_docs=True,
        )

    monkeypatch.setattr(api, "ask_docs", fake_ask_docs)

    response = client.post("/query", json={"question": "What does ERR_504 mean?"})
    assert response.status_code == 200
    data = response.json()
    assert data["found_in_docs"] is True
    assert data["citations"][0]["source"] == "errors.md"
    assert data["citations"][0]["page"] == 5