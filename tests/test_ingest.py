from pathlib import Path

import pytest
from langchain_core.documents import Document

import ingest


class FakeLoader:
    documents: list[Document] = []

    def __init__(self, path: str, *, glob: str, loader_cls: type) -> None:
        assert Path(path).is_absolute()
        assert glob == "**/*.txt"
        assert loader_cls is ingest.TextLoader

    def lazy_load(self):
        yield from self.documents


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
    FakeLoader.documents = [
        Document(page_content=f"Document {index}", metadata={"source": f"{index}.txt"})
        for index in range(3)
    ]
    collection = FakeCollection()
    monkeypatch.setattr(ingest, "DirectoryLoader", FakeLoader)
    monkeypatch.setattr(ingest, "setup_chroma_database", lambda: (object(), object()))
    monkeypatch.setattr(ingest, "Chroma", lambda **kwargs: collection)

    assert ingest.ingest_documents(root, batch_size=2) == 3
    assert [len(batch) for batch, _ in collection.batches] == [2, 1]
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
