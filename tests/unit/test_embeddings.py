from unittest.mock import patch

import numpy as np
import pandas as pd

from vector_bench.embeddings import (
    DEFAULT_MODEL_NAME,
    embed_corpus,
    passage_fn,
)


def test_passage_fn_prepends_title_to_description():
    assert passage_fn({"title": "Title", "description": "Description"}) == (
        "Title\n\nDescription"
    )


def test_passage_fn_returns_description_without_title():
    assert passage_fn({"description": "Description"}) == "Description"


@patch("vector_bench.embeddings.load_or_create_embeddings")
@patch("vector_bench.embeddings.ground_truth")
def test_embed_corpus_returns_ground_truth_and_embeddings(
    mock_ground_truth, mock_load_or_create_embeddings
):
    corpus = pd.DataFrame({"doc_id": ["doc-1"], "description": ["Description"]})
    judgments = pd.DataFrame(
        {"query_id": ["query-1"], "query": ["Query"], "doc_id": ["doc-1"]}
    )
    mock_load_or_create_embeddings.return_value = ([[0.1, 0.2]], object())
    mock_ground_truth.return_value = {"query-1": ["doc-1"]}

    result = embed_corpus(corpus, judgments, dataset_name="test")

    assert result[0] == {"query-1": ["doc-1"]}
    np.testing.assert_array_equal(
        result[1], np.array([[0.1, 0.2]], dtype=np.float32)
    )
    mock_load_or_create_embeddings.assert_called_once()
    mock_ground_truth.assert_called_once()
    assert mock_load_or_create_embeddings.call_args.kwargs["model_name"] == (
        DEFAULT_MODEL_NAME
    )


def test_ground_truth_uses_stable_dot_product_order_and_top_k(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(tmp_path))
    corpus = pd.DataFrame({"doc_id": ["doc-a", "doc-b", "doc-c"]})
    judgments = pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q2"],
            "query": ["first", "second", "second"],
        }
    )
    corpus_embeddings = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)

    with patch(
        "vector_bench.embeddings._query_embeddings",
        return_value=np.array([[1, 0], [0, 1]], dtype=np.float32),
    ):
        from vector_bench.embeddings import ground_truth

        result = ground_truth(
            corpus,
            judgments,
            corpus_embeddings,
            dataset_name="test",
            top_k=2,
        )

    assert result == {
        "q1": ["doc-a", "doc-c"],
        "q2": ["doc-b", "doc-c"],
    }


def test_ground_truth_uses_direct_dot_product_for_normalized_vectors(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(tmp_path))
    corpus = pd.DataFrame({"doc_id": ["doc-a", "doc-b"]})
    judgments = pd.DataFrame({"query_id": ["q1"], "query": ["query"]})
    corpus_embeddings = np.array([[1, 0], [2, 1]], dtype=np.float32)

    with patch(
        "vector_bench.embeddings._query_embeddings",
        return_value=np.array([[1, 0]], dtype=np.float32),
    ):
        from vector_bench.embeddings import ground_truth

        result = ground_truth(
            corpus,
            judgments,
            corpus_embeddings,
            dataset_name="test",
            top_k=2,
        )

    assert result == {"q1": ["doc-b", "doc-a"]}


def test_ground_truth_uses_direct_dot_product_with_zero_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(tmp_path))
    corpus = pd.DataFrame({"doc_id": ["doc-a"]})
    judgments = pd.DataFrame({"query_id": ["q1"], "query": ["query"]})

    with patch(
        "vector_bench.embeddings._query_embeddings",
        return_value=np.array([[1, 0]], dtype=np.float32),
    ):
        from vector_bench.embeddings import ground_truth

        result = ground_truth(
            corpus,
            judgments,
            np.array([[0, 0]], dtype=np.float32),
            dataset_name="test",
        )

    assert result == {"q1": ["doc-a"]}


def test_ground_truth_uses_cached_rankings(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(tmp_path))
    corpus = pd.DataFrame({"doc_id": ["doc-a", "doc-b"]})
    judgments = pd.DataFrame({"query_id": ["q1"], "query": ["query"]})
    corpus_embeddings = np.array([[1, 0], [0, 1]], dtype=np.float32)

    with patch(
        "vector_bench.embeddings._query_embeddings",
        return_value=np.array([[1, 0]], dtype=np.float32),
    ) as query_embeddings:
        from vector_bench.embeddings import ground_truth

        first_result = ground_truth(
            corpus,
            judgments,
            corpus_embeddings,
            dataset_name="test",
            top_k=1,
        )
        query_embeddings.assert_called_once()

    with patch("vector_bench.embeddings._query_embeddings") as query_embeddings:
        second_result = ground_truth(
            corpus,
            judgments,
            corpus_embeddings,
            dataset_name="test",
            top_k=1,
        )
        query_embeddings.assert_not_called()

    assert second_result == first_result
