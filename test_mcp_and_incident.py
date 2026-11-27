import asyncio
from fastapi.testclient import TestClient
import api
from mcp_server import mcp
from multi_agent import IncidentReport

client = TestClient(api.app)


def test_incident_endpoint(monkeypatch):
    def fake_pipeline(user_input: str, model: str = "ollama/qwen2.5:3b") -> IncidentReport:
        return IncidentReport(
            priority="P1",
            category="hybrid",
            root_cause="Upstream gateway timeout after 30s.",
            action_steps=["Check upstream health", "Inspect gateway logs"],
            citations=[{"source": "errors.md", "page": 5}],
        )

    monkeypatch.setattr(api, "run_incident_pipeline", fake_pipeline)

    response = client.post(
        "/incident",
        json={"description": "[2026-11-27] ERROR gateway: ERR_504. What does it mean?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["priority"] == "P1"
    assert data["category"] == "hybrid"
    assert data["citations"][0]["source"] == "errors.md"


def test_incident_validation_rejects_short_text():
    response = client.post("/incident", json={"description": "err"})
    assert response.status_code == 422


def test_mcp_server_registers_all_tools():
    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "search_docs_tool",
        "analyze_logs_tool",
        "resolve_incident_tool",
    }