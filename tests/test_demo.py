from pathlib import Path

from fastapi.testclient import TestClient

from demo import app
from ingest import ingest_documents

PROJECT_DATA = Path(__file__).parents[1] / "data"


def test_local_browser_demo_runs_real_retrieval_without_cloud(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("KAIRO_MODE", "local")
    monkeypatch.setenv("KAIRO_LOCAL_DB", str(tmp_path / "chroma"))
    monkeypatch.setenv("KAIRO_API_KEY", "test-local-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CHROMADB_API_KEY", raising=False)
    monkeypatch.delenv("TENANT_ID", raising=False)
    monkeypatch.delenv("DATABASE_ID", raising=False)
    ingest_documents(PROJECT_DATA)

    with TestClient(app) as client:
        page = client.get("/")
        response = client.post(
            "/query",
            headers={"Authorization": "Bearer test-local-key"},
            json={
                "question": "Who is Bernard?",
                "retrieval_mode": "similarity",
            },
        )

    assert page.status_code == 200
    assert "real local Chroma vector search" in page.text
    assert "test-local-key" in page.text
    assert response.status_code == 200
    assert "Bernard" in response.json()["answer"]
    assert response.json()["contexts"]
