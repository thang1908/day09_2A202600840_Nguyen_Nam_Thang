from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb

from rag.parser import parse_policy_markdown


class ChromaPolicyStore:
    """Student scaffold for the real Chroma-backed policy index."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: Any,
        collection_name: str = "policy_chunks",
    ) -> None:
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ensure_index(self, markdown_path: Path) -> None:
        if self.collection.count() == 0:
            self.rebuild(markdown_path)

    def rebuild(self, markdown_path: Path) -> None:
        markdown_text = markdown_path.read_text(encoding="utf-8")
        chunks = [_chunk_to_text(chunk) for chunk in parse_policy_markdown(markdown_text)]
        if not chunks:
            raise ValueError(f"No policy chunks parsed from {markdown_path}")

        self._reset_collection()
        embeddings = self.embedding_model.embed_documents(chunks)
        metadatas = [_metadata_for_chunk(chunk, index) for index, chunk in enumerate(chunks)]
        ids = [_chunk_id(chunk, index) for index, chunk in enumerate(chunks)]

        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        query = query.strip()
        if not query or self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=_normalize_top_k(top_k),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            hits.append(
                {
                    "citation": metadata.get("citation", _citation_from_chunk(document)),
                    "content": document,
                    "distance": distance,
                }
            )
        return hits

    def _reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def _metadata_for_chunk(chunk: str, index: int) -> dict[str, str | int]:
    section_h2, section_h3 = _sections_from_chunk(chunk)
    return {
        "chunk_index": index,
        "section_h2": section_h2,
        "section_h3": section_h3 or "",
        "citation": _citation_from_sections(section_h2, section_h3),
    }


def _chunk_to_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        rendered_text = chunk.get("rendered_text")
        if isinstance(rendered_text, str):
            return rendered_text
    raise TypeError(f"Unsupported policy chunk type: {type(chunk).__name__}")


def _chunk_id(chunk: str, index: int) -> str:
    digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]
    return f"policy-{index:04d}-{digest}"


def _citation_from_chunk(chunk: str) -> str:
    return _citation_from_sections(*_sections_from_chunk(chunk))


def _citation_from_sections(section_h2: str, section_h3: str | None) -> str:
    if section_h3:
        return f"{section_h2} > {section_h3}"
    return section_h2


def _sections_from_chunk(chunk: str) -> tuple[str, str | None]:
    section_h2 = ""
    section_h3: str | None = None

    for line in chunk.splitlines():
        if line.startswith("### ") and section_h3 is None:
            section_h3 = line.removeprefix("### ").strip()
        elif line.startswith("## ") and not section_h2:
            section_h2 = line.removeprefix("## ").strip()

    return section_h2, section_h3


def _normalize_top_k(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 4
