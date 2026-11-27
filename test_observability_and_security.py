from fastapi.testclient import TestClient
import api

client = TestClient(api.app)


def test_request_id_and_latency_headers():
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_metrics_endpoint_tracks_requests():
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] >= 1
    assert "/health" in data["by_path"]


def test_api_key_auth_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret-123")

    # /health әрқашан ашық қалуы керек
    assert client.get("/health").status_code == 200

    # Кілтсіз немесе қате кілтпен сұраныс 401 қайтаруы керек
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "wrong"}).status_code == 401

    # Дұрыс кілтпен сұраныс 200 қайтаруы керек
    ok = client.get("/metrics", headers={"X-API-Key": "test-secret-123"})
    assert ok.status_code == 200