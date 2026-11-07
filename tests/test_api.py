from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from agents_app.config import settings
from api import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "api-test.db"))
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "debug_responses", True)
    monkeypatch.setattr(settings, "api_keys", ())
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoints(client):
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json()["ready"] is True


def test_ui_and_static_assets_are_served(client):
    root = client.get("/")
    assert root.status_code == 200
    assert "Agentic Support" in root.text
    assert "app.js" in root.text
    assert "Content-Security-Policy" in root.headers

    script = client.get("/static/app.js")
    css = client.get("/static/app.css")
    assert script.status_code == 200
    assert "text/javascript" in script.headers["content-type"]
    assert css.status_code == 200


def test_chat_then_feedback_then_erasure(client):
    chat = client.post(
        "/v1/chat",
        json={"session_id": "session-1", "message": "Where is my order ORD-1003?"},
    )
    assert chat.status_code == 200
    payload = chat.json()
    assert payload["outcome"] == "answered"
    assert "ORD-1003" in payload["answer"]
    assert payload["debug"] is not None
    assert payload["debug"]["tool_calls"][0]["result"]["found"] is True

    feedback = client.post(
        "/v1/feedback", json={"trace_id": payload["trace_id"], "rating": 1}
    )
    assert feedback.status_code == 200
    assert feedback.json() == {"status": "recorded"}

    unknown = client.post(
        "/v1/feedback", json={"trace_id": "does-not-exist", "rating": -1}
    )
    assert unknown.status_code == 404

    erased = client.delete("/v1/sessions/session-1")
    assert erased.status_code == 200
    assert erased.json()["deleted_rows"] >= 1


def test_feedback_rejects_invalid_rating(client):
    response = client.post(
        "/v1/feedback", json={"trace_id": "x", "rating": 5}
    )
    assert response.status_code == 422


def test_chat_validation(client):
    bad_session = client.post(
        "/v1/chat", json={"session_id": "bad id!", "message": "hello"}
    )
    empty_message = client.post(
        "/v1/chat", json={"session_id": "ok", "message": ""}
    )
    assert bad_session.status_code == 422
    assert empty_message.status_code == 422


def test_streaming_returns_steps_then_an_answer(client):
    with client.stream(
        "POST",
        "/v1/chat/stream",
        json={"session_id": "stream-1", "message": "My router keeps disconnecting"},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = "".join(response.iter_text())

    assert "event: step" in body
    assert "event: answer" in body
    answer_line = next(
        line for line in body.splitlines() if line.startswith("data: ") and '"trace_id"' in line
    )
    payload = json.loads(answer_line[len("data: "):])
    assert payload["outcome"] == "answered"
    assert payload["verified"] is True


def test_metrics_endpoint_exposes_prometheus_text(client):
    client.post(
        "/v1/chat", json={"session_id": "metric-1", "message": "My router keeps disconnecting"}
    )
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "support_turns_total" in response.text
    assert "support_outcomes_total" in response.text


def test_oversized_body_is_rejected_before_parsing(client):
    response = client.post(
        "/v1/chat",
        content=json.dumps({"session_id": "big", "message": "x" * 70_000}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json() == {"error": "payload_too_large"}


def test_invalid_customer_header_is_rejected(client):
    response = client.post(
        "/v1/chat",
        json={"session_id": "cust-1", "message": "hello"},
        headers={"X-Customer-Id": "bad customer id!"},
    )
    assert response.status_code == 422
