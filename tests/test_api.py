from fastapi.testclient import TestClient

from api import create_app
from rag import RAGResponse


class FakeRAGService:
    def __init__(self) -> None:
        self.arguments: tuple[str, str, int] | None = None

    def ask(self, question: str, retrieval_mode: str, k: int) -> RAGResponse:
        self.arguments = (question, retrieval_mode, k)
        return RAGResponse(
            answer="A grounded answer.",
            contexts=["Relevant passage."],
            latency_ms=12.5,
            retrieval_mode=retrieval_mode,
        )


def test_query_selects_variant_and_returns_measurements() -> None:
    service = FakeRAGService()
    app = create_app(lambda: service)

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "question": "  What happened?  ",
                "retrieval_mode": "similarity",
                "k": 5,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "A grounded answer.",
        "contexts": ["Relevant passage."],
        "latency_ms": 12.5,
        "retrieval_mode": "similarity",
    }
    assert service.arguments == ("What happened?", "similarity", 5)


def test_query_rejects_blank_question_and_invalid_variant() -> None:
    app = create_app(FakeRAGService)

    with TestClient(app) as client:
        blank = client.post("/query", json={"question": "  "})
        invalid_variant = client.post(
            "/query",
            json={"question": "question", "retrieval_mode": "random"},
        )

    assert blank.status_code == 422
    assert invalid_variant.status_code == 422
