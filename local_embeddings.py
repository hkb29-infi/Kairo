import hashlib
import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class LocalHashEmbeddings(Embeddings):
    """Deterministic lexical embeddings that run without a model or API."""

    def __init__(self, dimensions: int = 1024) -> None:
        if dimensions < 64:
            raise ValueError("dimensions must be at least 64.")
        self._dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        tokens = TOKEN_PATTERN.findall(text.casefold())
        features: Counter[str] = Counter(f"word:{token}" for token in tokens)
        features.update(
            f"pair:{left}_{right}" for left, right in zip(tokens, tokens[1:])
        )
        for token in tokens:
            if len(token) >= 3:
                features.update(
                    f"char:{token[index:index + width]}"
                    for width in (3, 4)
                    for index in range(len(token) - width + 1)
                )

        vector = [0.0] * self._dimensions
        for feature, weight in features.items():
            digest = hashlib.blake2b(
                feature.encode("utf-8"), digest_size=8
            ).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * weight

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector
