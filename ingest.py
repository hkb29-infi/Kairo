# ============================================================
# ChromaDB Cloud Setup & Document Ingestion Module
# ------------------------------------------------------------
# This module sets up a connection to ChromaDB Cloud, initializes
# Google Generative AI embeddings, and ingests local text files
# into a Chroma collection for retrieval-augmented generation (RAG).
# ============================================================

import os
import hashlib
from pathlib import Path
from typing import Literal

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

from local_embeddings import LocalHashEmbeddings

load_dotenv()

collection_name = "my-rag-collection"
StorageMode = Literal["local", "cloud"]
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def get_storage_mode(mode: StorageMode | None = None) -> StorageMode:
    selected_mode = mode or os.getenv("KAIRO_MODE", "local").strip().lower()
    if selected_mode not in ("local", "cloud"):
        raise ValueError("KAIRO_MODE must be either 'local' or 'cloud'.")
    return selected_mode


def setup_chroma_database() -> tuple[chromadb.ClientAPI, Embeddings]:
    """Connect to Chroma Cloud using Gemini embeddings."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    client = chromadb.CloudClient(
        api_key=os.environ["CHROMADB_API_KEY"],
        tenant=os.environ["TENANT_ID"],
        database=os.environ["DATABASE_ID"],
    )
    return client, embeddings


def setup_vector_store(mode: StorageMode | None = None) -> Chroma:
    selected_mode = get_storage_mode(mode)
    if selected_mode == "local":
        persist_directory = Path(
            os.getenv("KAIRO_LOCAL_DB", ".kairo/chroma")
        ).expanduser()
        persist_directory.mkdir(parents=True, exist_ok=True)
        client: chromadb.ClientAPI = chromadb.PersistentClient(
            path=str(persist_directory.resolve())
        )
        embeddings = LocalHashEmbeddings()
    else:
        client, embeddings = setup_chroma_database()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=client,
    )


def _document_id(document: Document, chunk_index: int) -> str:
    source = str(document.metadata["source"]).replace("\\", "/")
    identity = (
        f"{source}\0{chunk_index}\0{document.page_content}".encode("utf-8")
    )
    return hashlib.sha256(identity).hexdigest()


def _chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start + chunk_size // 2, end),
                text.rfind(". ", start + chunk_size // 2, end),
                text.rfind(" ", start + chunk_size // 2, end),
            )
            if boundary > start:
                end = boundary + (2 if text[boundary:boundary + 2] == ". " else 0)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - chunk_overlap)
    return chunks


def ingest_documents(
    data_directory: str | Path = "./data",
    batch_size: int = 100,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> int:
    """Chunk local TXT files and upsert them into the configured Chroma store."""
    root = Path(data_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")

    collection = setup_vector_store()
    batch: list[Document] = []
    ids: list[str] = []
    ingested_count = 0
    for source_path in sorted(root.rglob("*.txt")):
        relative_source = source_path.relative_to(root).as_posix()
        text = source_path.read_text(encoding="utf-8")
        for chunk_index, chunk in enumerate(
            _chunk_text(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ):
            document = Document(
                page_content=chunk,
                metadata={"source": relative_source, "chunk_index": chunk_index},
            )
            batch.append(document)
            ids.append(_document_id(document, chunk_index))
            if len(batch) >= batch_size:
                collection.add_documents(batch, ids=ids)
                ingested_count += len(batch)
                batch, ids = [], []

    if batch:
        collection.add_documents(batch, ids=ids)
        ingested_count += len(batch)
    if ingested_count == 0:
        raise FileNotFoundError(f"No non-empty .txt documents found in {root}")
    return ingested_count


if __name__ == "__main__":
    count = ingest_documents()
    print(
        f"Upserted {count} document chunks into '{collection_name}' "
        f"({get_storage_mode()} mode)."
    )
