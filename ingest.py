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
from typing import Tuple
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from dotenv import load_dotenv


load_dotenv()

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
collection_name = "my-rag-collection"


# -------------------------------------------------------------
# Database setup
# -------------------------------------------------------------
def setup_chroma_database() -> Tuple[chromadb.ClientAPI, GoogleGenerativeAIEmbeddings]:
    """Set up and authenticate ChromaDB Cloud connection.

    Initializes the Google Generative AI embeddings model and connects to
    the ChromaDB Cloud service using credentials stored in environment variables.

    Returns:
        Tuple[chromadb.ClientAPI, GoogleGenerativeAIEmbeddings]:
            A tuple containing:
            - client: Authenticated ChromaDB Cloud client.
            - embeddings: GoogleGenerativeAIEmbeddings instance.
    """
    try: 
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"]
            )
        client = chromadb.CloudClient(
            api_key=os.environ["CHROMADB_API_KEY"],
            tenant=os.environ["TENANT_ID"],
            database=os.environ["DATABASE_ID"]
            )
        return client, embeddings
    except Exception as e:
        print(f"Failed to initialize ChromaDB Cloud Client: {e}")
        raise


# -------------------------------------------------------------
# Document Ingestion
# -------------------------------------------------------------
def _document_id(document: Document) -> str:
    source = str(document.metadata["source"]).replace("\\", "/")
    identity = f"{source}\0{document.page_content}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def ingest_documents(
    data_directory: str | Path = "./data",
    batch_size: int = 100,
) -> int:
    """Stream text files into Chroma in bounded, repeatable batches."""
    root = Path(data_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    client, embeddings = setup_chroma_database()
    collection = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=client,
    )
    loader = DirectoryLoader(
        str(root), glob="**/*.txt", loader_cls=TextLoader
    )

    batch: list[Document] = []
    ids: list[str] = []
    ingested_count = 0
    for document in loader.lazy_load():
        source_path = Path(document.metadata["source"])
        if not source_path.is_absolute():
            source_path = root / source_path
        document.metadata["source"] = source_path.resolve().relative_to(root).as_posix()
        batch.append(document)
        ids.append(_document_id(document))
        if len(batch) == batch_size:
            collection.add_documents(batch, ids=ids)
            ingested_count += len(batch)
            batch, ids = [], []

    if batch:
        collection.add_documents(batch, ids=ids)
        ingested_count += len(batch)
    if ingested_count == 0:
        raise FileNotFoundError(f"No .txt documents found in {root}")
    return ingested_count


if __name__ == "__main__":
    count = ingest_documents()
    print(f"Ingested {count} documents into collection '{collection_name}'.")
