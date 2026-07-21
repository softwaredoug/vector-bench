"""Create embeddings for vector-bench corpora."""

from collections.abc import Iterator
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
) -> list[str]:
    """Create or load embeddings and return student-tool CSV lines."""
    embeddings, _model = load_or_create_embeddings(
        corpus,
        passage_fn=passage_fn,
        model_name=model_name,
        device=device,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )
    return list(embedding_csv_lines(corpus, embeddings))


def embedding_csv_lines(corpus: Any, embeddings: Any) -> Iterator[str]:
    """Yield student-tool CSV lines for corpus embeddings."""
    if len(corpus) != len(embeddings):
        raise ValueError("Corpus and embeddings must contain the same number of rows")

    for doc_id, embedding in zip(corpus["doc_id"], embeddings):
        values = ",".join(str(value) for value in embedding)
        yield f"{doc_id},{values}\n"
