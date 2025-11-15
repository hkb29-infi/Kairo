from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from rag import RAGService


class FakeRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.query: str | None = None

    def invoke(self, query: str) -> list[Document]:
        self.query = query
        return self.documents


class FakeVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.search_type: str | None = None
        self.search_kwargs: dict[str, int] | None = None

    def as_retriever(
        self, *, search_type: str, search_kwargs: dict[str, int]
    ) -> FakeRetriever:
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        return FakeRetriever(self.documents)


class FakeLLM:
    def __init__(self) -> None:
        self.prompt: str | None = None

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompt = prompt
        return SimpleNamespace(content="A grounded answer.")


def test_ask_uses_selected_retrieval_and_returns_contexts() -> None:
    store = FakeVectorStore([Document(page_content="Source passage")])
    llm = FakeLLM()

    result = RAGService(llm, store).ask(
        "  What is in the source?  ", retrieval_mode="similarity", k=5
    )

    assert result.answer == "A grounded answer."
    assert result.contexts == ["Source passage"]
    assert result.retrieval_mode == "similarity"
    assert result.latency_ms >= 0
    assert store.search_type == "similarity"
    assert store.search_kwargs == {"k": 5}
    assert "What is in the source?" in llm.prompt
    assert "Source passage" in llm.prompt


@pytest.mark.parametrize(
    ("question", "mode", "k", "message"),
    [
        ("  ", "mmr", 3, "Question must not be empty."),
        ("question", "unsupported", 3, "Unsupported retrieval mode"),
        ("question", "mmr", 0, "k must be at least 1."),
    ],
)
def test_ask_rejects_invalid_input(
    question: str, mode: str, k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RAGService(FakeLLM(), FakeVectorStore([])).ask(
            question, retrieval_mode=mode, k=k
        )
