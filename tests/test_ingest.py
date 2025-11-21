from pathlib import Path

import pytest
from langchain_core.documents import Document

import ingest


class FakeCollection:
    def __init__(self) -> None:
        self.batches: list[tuple[list[Document], list[str]]] = []

    def add_documents(self, documents: list[Document], *, ids: list[str]) -> None:
        self.batches.append((documents, ids))


def test_ingestion_streams_batches_with_stable_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    for index in range(3):
        (root / f"{index}.txt").write_text(f"Document {index}", encoding="utf-8")
    collection = FakeCollection()
    monkeypatch.setattr(ingest, "setup_vector_store", lambda: collection)

    assert ingest.ingest_documents(root, batch_size=2) == 3
    assert [len(batch) for batch, _ in collection.batches] == [2, 1]
    assert all("chunk_index" in doc.metadata for batch, _ in collection.batches for doc in batch)
    original_ids = [doc_id for _, batch_ids in collection.batches for doc_id in batch_ids]

    collection.batches.clear()
    assert ingest.ingest_documents(root, batch_size=2) == 3
    repeated_ids = [doc_id for _, batch_ids in collection.batches for doc_id in batch_ids]
    assert repeated_ids == original_ids


def test_ingestion_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Document directory does not exist"):
        ingest.ingest_documents(tmp_path / "missing")


def test_ingestion_rejects_invalid_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        ingest.ingest_documents(tmp_path, batch_size=0)


def test_ingestion_splits_long_documents_with_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "guide.txt").write_text(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        encoding="utf-8",
    )
    collection = FakeCollection()
    monkeypatch.setattr(ingest, "setup_vector_store", lambda: collection)

    assert ingest.ingest_documents(root, chunk_size=24, chunk_overlap=5) > 1
    chunks = [doc for batch, _ in collection.batches for doc in batch]
    assert len({doc.metadata["chunk_index"] for doc in chunks}) == len(chunks)
    text_chunks = ingest._chunk_text("abcdefghij klmnop", chunk_size=10, chunk_overlap=3)
    assert text_chunks[0][-3:] == text_chunks[1][:3]
