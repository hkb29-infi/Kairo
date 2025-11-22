import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from ingest import get_storage_mode, setup_vector_store

# the RAG pipeline to make the retrieved answers better match the grounded truth we provide
RetrievalMode = Literal["similarity", "mmr"]
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "of", "on",
    "or", "our", "the", "this", "to", "us", "was", "what", "when",
    "where", "which", "who", "why", "with", "you", "your",
}

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
    def __init__(
        self, llm: BaseLanguageModel | None, vector_store: Chroma
    ) -> None:
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
        if self._llm is None:
            answer = _extractive_answer(question, contexts)
            latency_ms = (perf_counter() - started_at) * 1000
            return RAGResponse(
                answer=answer,
                contexts=contexts,
                latency_ms=latency_ms,
                retrieval_mode=retrieval_mode,
            )

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


def _extractive_answer(question: str, contexts: list[str]) -> str:
    query_terms = {
        token
        for token in TOKEN_PATTERN.findall(question.casefold())
        if token not in STOP_WORDS
    }
    if not query_terms:
        return "I couldn't find a supported answer in the indexed documents."

    ranked_sentences: list[tuple[int, str]] = []
    for context in contexts:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", context):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_terms = set(TOKEN_PATTERN.findall(sentence.casefold()))
            overlap = len(query_terms & sentence_terms)
            if overlap:
                ranked_sentences.append((overlap, sentence))

    if not ranked_sentences:
        return "I couldn't find a supported answer in the retrieved documents."
    ranked_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = [sentence for _, sentence in ranked_sentences[:3]]
    return " ".join(dict.fromkeys(selected))


def create_rag_service() -> RAGService:
    mode = get_storage_mode()
    vector_store = setup_vector_store(mode)
    llm = None
    if mode == "cloud":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0,
            api_key=os.environ["GOOGLE_API_KEY"],
        )
    return RAGService(llm, vector_store)
