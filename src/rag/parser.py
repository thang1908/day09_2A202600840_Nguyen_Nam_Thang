from __future__ import annotations

from langchain_text_splitters import MarkdownHeaderTextSplitter


def parse_policy_markdown(markdown_text: str) -> list[str]:
    """Split policy markdown into rendered markdown text chunks by H2/H3."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("##", "section_h2"),
            ("###", "section_h3"),
        ],
        strip_headers=True,
    )

    chunks: list[str] = []
    for document in splitter.split_text(markdown_text):
        section_h2 = document.metadata.get("section_h2")
        if not section_h2:
            continue

        chunks.append(
            _render_chunk_text(
                section_h2=section_h2,
                section_h3=document.metadata.get("section_h3"),
                content=document.page_content,
            )
        )

    return chunks


def _render_chunk_text(
    section_h2: str,
    section_h3: str | None,
    content: str,
) -> str:
    parts = [f"## {section_h2}"]
    if section_h3:
        parts.append(f"### {section_h3}")
    if content:
        parts.append(content)
    return "\n\n".join(parts)
