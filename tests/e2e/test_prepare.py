import h5py
import numpy as np
import pytest

from vector_bench.datasets import get_dataset
from vector_bench.prepare import main


@pytest.mark.parametrize("num_queries", [None, 2])
def test_prepare_main_writes_index_and_queries(
    tmp_path, monkeypatch, num_queries
):
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(tmp_path / "cache"))
    index_path = tmp_path / "index.h5"
    queries_path = tmp_path / "queries.h5"

    arguments = [
        "--dataset",
        "dougs_blog_data",
        "--top-k",
        "5",
        "--index-out",
        str(index_path),
        "--queries-out",
        str(queries_path),
    ]
    if num_queries is not None:
        arguments[4:4] = ["--num-queries", str(num_queries)]
    main(arguments)

    corpus, judgments = get_dataset("dougs_blog_data")
    with h5py.File(index_path) as index_file:
        doc_ids = [value.decode() if isinstance(value, bytes) else value for value in index_file["doc_ids"]]
        document_vectors = index_file["vectors"][:]
    with h5py.File(queries_path) as queries_file:
        query_file_ids = [value.decode() if isinstance(value, bytes) else value for value in queries_file["query_ids"]]
        query_vectors = queries_file["vectors"][:]
        ground_truth = queries_file["ground_truth"][:]

    assert len(doc_ids) == len(corpus)
    assert doc_ids[0] == str(corpus.iloc[0]["doc_id"])
    assert document_vectors.ndim == 2

    query_ids = [str(query_id) for query_id in judgments["query_id"].drop_duplicates()]
    if num_queries is not None:
        query_ids = query_ids[:num_queries]
    assert query_file_ids == sorted(query_ids)
    assert query_vectors.shape[0] == len(query_ids)
    assert ground_truth.shape[0] == len(query_ids)

    for query_index, query_id in enumerate(query_file_ids):
        ranked_doc_ids = [
            value.decode() if isinstance(value, bytes) else value
            for value in ground_truth[query_index]
            if value not in (b"", "")
        ]
        ranked_vectors = np.asarray(
            [document_vectors[doc_ids.index(doc_id)] for doc_id in ranked_doc_ids]
        )
        expected_order = np.argsort(
            -(ranked_vectors @ query_vectors[query_index]), kind="stable"
        )
        assert [ranked_doc_ids[index] for index in expected_order] == ranked_doc_ids


def test_wands_dataset_is_available():
    corpus, judgments = get_dataset("wands")

    assert len(corpus) > 40_000
    assert len(judgments) > 200_000
    assert {"doc_id", "title", "description"}.issubset(corpus.columns)
    assert {"query_id", "doc_id"}.issubset(judgments.columns)
