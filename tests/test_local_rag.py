from pathlib import Path

import pytest

from ingest import get_storage_mode, ingest_documents
from local_embeddings import LocalHashEmbeddings
from rag import create_rag_service


def test_local_embeddings_are_deterministic_and_normalized() -> None:
    embeddings = LocalHashEmbeddings()
    first = embeddings.embed_query("local document retrieval")
    repeated = embeddings.embed_query("local document retrieval")
    different = embeddings.embed_query("unrelated vocabulary")

    assert len(first) == 1024
    assert first == repeated
    assert sum(value * value for value in first) == pytest.approx(1.0)
    assert first != different


def test_local_store_ingestion_and_query_work_without_cloud_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "handbook.txt").write_text(
        "Vacation leave requests must be submitted to the manager by email. "
        "Managers approve vacation leave requests before time off begins. "
        "Medical leave follows a separate confidential process.",
        encoding="utf-8",
    )
    monkeypatch.setenv("KAIRO_MODE", "local")
    monkeypatch.setenv("KAIRO_LOCAL_DB", str(tmp_path / "chroma"))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CHROMADB_API_KEY", raising=False)
    monkeypatch.delenv("TENANT_ID", raising=False)
    monkeypatch.delenv("DATABASE_ID", raising=False)

    assert get_storage_mode() == "local"
    assert ingest_documents(data, chunk_size=160, chunk_overlap=20) >= 1
    service = create_rag_service()

    result = service.ask("How does the manager approve vacation leave?")

    assert result.contexts
    assert any("manager" in context.casefold() for context in result.contexts)
    assert "manager" in result.answer.casefold()
    assert result.latency_ms >= 0
    assert result.retrieval_mode == "mmr"


def test_invalid_storage_mode_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIRO_MODE", "unknown")
    with pytest.raises(ValueError, match="KAIRO_MODE must be either"):
        get_storage_mode()
