import csv

import numpy as np
import pytest

from vector_bench.datasets import get_dataset
from vector_bench.prepare import main


@pytest.mark.parametrize("num_queries", [None, 2])
def test_prepare_main_writes_index_and_queries(
    tmp_path, monkeypatch, num_queries
):
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(tmp_path / "cache"))
    index_path = tmp_path / "index.csv"
    queries_path = tmp_path / "queries.csv"

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
    with index_path.open(newline="") as index_file:
        index_rows = list(csv.reader(index_file))
    with queries_path.open(newline="") as queries_file:
        query_rows = list(csv.reader(queries_file))

    assert len(index_rows) == len(corpus)
    assert index_rows[0][0] == str(corpus.iloc[0]["doc_id"])
    assert len(index_rows[0]) > 1

    query_ids = [str(query_id) for query_id in judgments["query_id"].drop_duplicates()]
    if num_queries is not None:
        query_ids = query_ids[:num_queries]
    query_embedding_ids = [
        row[0] for row in query_rows if row[1:3] == ["-1", "-1"]
    ]
    assert query_embedding_ids == sorted(query_ids)
    assert {row[0] for row in query_rows} == set(query_ids)
    assert all(len(row) == len(index_rows[0]) + 2 for row in query_rows)

    rows_by_query = {}
    for row in query_rows:
        rows_by_query.setdefault(row[0], []).append(row)
    for rows in rows_by_query.values():
        assert rows[0][1:3] == ["-1", "-1"]
        assert [int(row[2]) for row in rows[1:]] == list(range(1, len(rows)))
        assert rows == sorted(rows, key=lambda row: int(row[2]))

        query_vector = np.asarray([float(value) for value in rows[0][3:]])
        document_vectors = np.asarray(
            [[float(value) for value in row[3:]] for row in rows[1:]]
        )
        expected_order = np.argsort(-(document_vectors @ query_vector), kind="stable")
        assert [rows[index + 1][1] for index in expected_order] == [
            row[1] for row in rows[1:]
        ]
