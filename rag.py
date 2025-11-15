import os
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from ingest import collection_name, setup_chroma_database

RetrievalMode = Literal["similarity", "mmr"]

SYSTEM_PROMPT = """You are a helpful AI assistant. Answer using only the
provided context. If the context does not contain the answer, say that you
don't know. Do not invent facts.

Context:
{context}

Question: {question}

Answer:"""


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    contexts: list[str]
    latency_ms: float
    retrieval_mode: RetrievalMode


class RAGService:
    def __init__(self, llm: ChatGoogleGenerativeAI, vector_store: Chroma) -> None:
        self._llm = llm
        self._vector_store = vector_store

    def ask(
        self,
        question: str,
        retrieval_mode: RetrievalMode = "mmr",
        k: int = 3,
    ) -> RAGResponse:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if retrieval_mode not in ("similarity", "mmr"):
            raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
        if k < 1:
            raise ValueError("k must be at least 1.")

        started_at = perf_counter()
        retriever = self._vector_store.as_retriever(
            search_type=retrieval_mode,
            search_kwargs={"k": k},
        )
        documents: list[Document] = retriever.invoke(question)
        contexts = [document.page_content for document in documents]
        context = "\n\n".join(contexts)
        response = self._llm.invoke(
            SYSTEM_PROMPT.format(context=context, question=question)
        )
        latency_ms = (perf_counter() - started_at) * 1000
        return RAGResponse(
            answer=str(response.content),
            contexts=contexts,
            latency_ms=latency_ms,
            retrieval_mode=retrieval_mode,
        )


def create_rag_service() -> RAGService:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0,
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    client: chromadb.ClientAPI
    embeddings: GoogleGenerativeAIEmbeddings
    client, embeddings = setup_chroma_database()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=client,
    )
    return RAGService(llm, vector_store)
