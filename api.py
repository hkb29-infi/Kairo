import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Callable, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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
    api_key: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_dotenv()
        configured_api_key = (
            api_key if api_key is not None else os.getenv("KAIRO_API_KEY")
        )
        if not configured_api_key:
            raise RuntimeError("KAIRO_API_KEY must be configured for the HTTP API.")
        app.state.api_key = configured_api_key
        app.state.rag_service = service_factory()
        try:
            yield
        finally:
            app.state.rag_service = None

    app = FastAPI(title="Kairo RAG API", lifespan=lifespan)
    bearer_scheme = HTTPBearer(auto_error=False)

    def require_api_key(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ) -> None:
        expected_api_key = getattr(app.state, "api_key", None)
        if (
            expected_api_key is None
            or credentials is None
            or not secrets.compare_digest(credentials.credentials, expected_api_key)
        ):
            raise HTTPException(
                status_code=401,
                detail="A valid bearer API key is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.post(
        "/query",
        response_model=QueryResponse,
        dependencies=[Depends(require_api_key)],
    )
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
