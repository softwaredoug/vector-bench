"""Create embeddings for vector-bench corpora."""

from typing import Any

from cheat_at_search.embeddings import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    load_or_create_embeddings,
)


def passage_fn(row: Any) -> str:
    """Build the text used to embed one corpus row."""
    title = row.get("title")
    description = row.get("description", "")

    if title:
        return f"{title}\n\n{description}"
    return description


def embed_corpus(
    corpus: Any,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    show_progress: bool = True,
) -> tuple[Any, Any]:
    """Create or load embeddings for a corpus."""
    return load_or_create_embeddings(
        corpus,
        passage_fn=passage_fn,
        model_name=model_name,
        device=device,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )
