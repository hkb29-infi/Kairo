from fastapi.testclient import TestClient

from demo import DEMO_API_KEY, OfflineDemoService, app


def test_offline_demo_returns_sample_context_and_explicit_disclaimer() -> None:
    result = OfflineDemoService().ask(
        "What does the sample say?", retrieval_mode="similarity"
    )

    assert result.retrieval_mode == "similarity"
    assert result.contexts
    assert "fixed offline-demo response" in result.answer
    assert result.latency_ms >= 0


def test_offline_demo_page_and_query_endpoint_are_available() -> None:
    with TestClient(app) as client:
        page = client.get("/")
        response = client.post(
            "/query",
            headers={"Authorization": f"Bearer {DEMO_API_KEY}"},
            json={"question": "What does the document say?"},
        )

    assert page.status_code == 200
    assert "Offline demo, not the live Gemini/Chroma system." in page.text
    assert response.status_code == 200
    assert response.json()["contexts"]
    assert "no LLM was called" in response.json()["answer"]
