import numpy as np

from vector_bench.naive_search import VectorIndex


def test_index_keeps_document_ids_and_first_20_dimensions():
    rows = [
        ["doc-a", "1", "2"] + ["0"] * 19 + ["999"],
    ]

    index = VectorIndex.index(
        [row[0] for row in rows],
        np.asarray([[float(value) for value in rows[0][1:]]]),
    )

    assert index.doc_ids == ["doc-a"]
    assert index.doc_vectors.shape == (1, 20)
    assert index.doc_vectors[0, :2].tolist() == [1, 2]


def test_query_returns_documents_in_dot_product_order():
    index = VectorIndex.index(
        ["doc-a", "doc-b", "doc-c"],
        np.array(
            [
                [1, 0] + [0] * 18,
                [0, 2] + [0] * 18,
                [0, 0] + [0] * 18,
            ],
            dtype=np.float32,
        ),
    )

    results = index.query(np.array([1, 1] + [0] * 18))

    assert results == [(1, "doc-b"), (2, "doc-a"), (3, "doc-c")]
