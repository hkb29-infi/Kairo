import logging
from contextlib import asynccontextmanager
from typing import Callable, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from rag import RAGResponse, RAGService, create_rag_service

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    retrieval_mode: Literal["similarity", "mmr"] = "mmr"
    k: int = Field(default=3, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be blank.")
        return question


class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]
    latency_ms: float
    retrieval_mode: Literal["similarity", "mmr"]

    @classmethod
    def from_result(cls, result: RAGResponse) -> "QueryResponse":
        return cls(
            answer=result.answer,
            contexts=result.contexts,
            latency_ms=result.latency_ms,
            retrieval_mode=result.retrieval_mode,
        )


def create_app(
    service_factory: Callable[[], RAGService] = create_rag_service,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_dotenv()
        app.state.rag_service = service_factory()
        try:
            yield
        finally:
            app.state.rag_service = None

    app = FastAPI(title="Kairo RAG API", lifespan=lifespan)

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        service: RAGService = request.app.state.rag_service
        try:
            result = service.ask(
                payload.question,
                retrieval_mode=payload.retrieval_mode,
                k=payload.k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("RAG query failed")
            raise HTTPException(
                status_code=502,
                detail="RAG backend request failed.",
            ) from exc
        return QueryResponse.from_result(result)

    return app


app = create_app()
