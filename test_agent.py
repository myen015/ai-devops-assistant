from fastapi.testclient import TestClient
import api
from agent import AgentResponse, ToolCallRecord

client = TestClient(api.app)


def test_agent_endpoint(monkeypatch):
    def fake_run_agent(
        user_prompt: str,
        model: str = "ollama_chat/qwen2.5:3b",
        max_steps: int = 4,
        session_id: str | None = None,
    ) -> AgentResponse:
        return AgentResponse(
            answer="ERR_504 means upstream gateway timeout after 30 seconds.",
            tools_used=[ToolCallRecord(tool="search_documentation", args={"question": "ERR_504"})],
            steps_taken=2,
            session_id=session_id,
        )

    monkeypatch.setattr(api, "run_agent", fake_run_agent)

    response = client.post(
        "/agent",
        json={"prompt": "What is ERR_504?", "session_id": "test-session"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["steps_taken"] == 2
    assert data["session_id"] == "test-session"
    assert data["tools_used"][0]["tool"] == "search_documentation"


def test_agent_prompt_validation():
    response = client.post("/agent", json={"prompt": "hi"})
    assert response.status_code == 422


def test_sources_endpoint(monkeypatch):
    monkeypatch.setattr(
        api,
        "list_sources",
        lambda: [{"source": "errors.md", "chunks": 1, "pages": [5]}],
    )
    response = client.get("/sources")
    assert response.status_code == 200
    assert response.json()["sources"][0]["source"] == "errors.md"