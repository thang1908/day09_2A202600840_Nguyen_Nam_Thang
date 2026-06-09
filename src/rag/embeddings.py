from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbeddingModel:
    """Small wrapper exposing the same methods as SentenceTransformerEmbeddings."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = OpenAIEmbeddings(model=model_name, api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)


def build_embedding_model(
    model_name: str = "text-embedding-3-small",
    api_key: str | None = None,
) -> OpenAIEmbeddingModel | SentenceTransformerEmbeddings:
    if _is_openai_embedding_model(model_name):
        return OpenAIEmbeddingModel(
            model_name=_normalize_openai_model_name(model_name),
            api_key=api_key,
        )
    return SentenceTransformerEmbeddings(model_name)


def _is_openai_embedding_model(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith("text-embedding-") or normalized.startswith("openai:")


def _normalize_openai_model_name(model_name: str) -> str:
    if model_name.lower().startswith("openai:"):
        return model_name.split(":", 1)[1]
    return model_name
