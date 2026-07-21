from unittest.mock import patch

import pandas as pd
import pytest

from vector_bench.embeddings import (
    DEFAULT_MODEL_NAME,
    embed_corpus,
    embedding_csv_lines,
    passage_fn,
)


def test_passage_fn_prepends_title_to_description():
    assert passage_fn({"title": "Title", "description": "Description"}) == (
        "Title\n\nDescription"
    )


def test_passage_fn_returns_description_without_title():
    assert passage_fn({"description": "Description"}) == "Description"


def test_embedding_csv_lines_serializes_doc_ids_and_vectors():
    corpus = pd.DataFrame({"doc_id": ["doc-1", "doc-2"]})
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    assert list(embedding_csv_lines(corpus, embeddings)) == [
        "doc-1,0.1,0.2\n",
        "doc-2,0.3,0.4\n",
    ]


def test_embedding_csv_lines_rejects_mismatched_row_counts():
    corpus = pd.DataFrame({"doc_id": ["doc-1"]})

    with pytest.raises(ValueError, match="same number of rows"):
        list(embedding_csv_lines(corpus, [[0.1], [0.2]]))


@patch("vector_bench.embeddings.load_or_create_embeddings")
def test_embed_corpus_returns_csv_lines(mock_load_or_create_embeddings):
    corpus = pd.DataFrame({"doc_id": ["doc-1"], "description": ["Description"]})
    mock_load_or_create_embeddings.return_value = ([[0.1, 0.2]], object())

    result = embed_corpus(corpus)

    assert result == ["doc-1,0.1,0.2\n"]
    mock_load_or_create_embeddings.assert_called_once()
    assert mock_load_or_create_embeddings.call_args.kwargs["model_name"] == (
        DEFAULT_MODEL_NAME
    )
