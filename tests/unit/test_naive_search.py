import numpy as np
from unittest.mock import patch

from exps.naive import VectorIndex
import h5py


def index_of(tmp_path, doc_ids, vectors):
    index_path = tmp_path / "embeddings.h5"
    with h5py.File(index_path, "w") as f:
        f.create_dataset("doc_ids", data=[doc_id.encode() for doc_id in doc_ids])
        f.create_dataset("vectors", data=vectors)
    with h5py.File(index_path, "r") as f:
        doc_ids_dataset = f["doc_ids"]
        vectors_dataset = f["vectors"]
        if not isinstance(doc_ids_dataset, h5py.Dataset) or not isinstance(vectors_dataset, h5py.Dataset):
            raise ValueError("Expected 'doc_ids' and 'vectors' datasets in HDF5 file")
        index = VectorIndex.index(doc_ids_dataset, vectors_dataset)
        return index


def test_index_keeps_document_ids_and_all_dimensions(tmp_path):
    doc_ids = ["doc-a"]
    vectors = np.array([[1, 2] + [0] * 19 + [999]], dtype=np.float32)

    index = index_of(tmp_path, doc_ids, vectors)

    assert index.doc_ids == ["doc-a"]
    assert index.doc_vectors.shape == (1, 22)
    assert index.doc_vectors[0, :2].tolist() == [1, 2]
    assert index.doc_vectors[0, -1] == 999


def test_query_returns_documents_in_dot_product_order(tmp_path):
    doc_ids = ["doc-a", "doc-b", "doc-c"]
    vectors = np.array(
        [
            [1, 0] + [0] * 18,
            [0, 2] + [0] * 18,
            [0, 0] + [0] * 18,
        ],
        dtype=np.float32,
    )

    index = index_of(tmp_path, doc_ids, vectors)
    results = index.query(np.array([1, 1] + [0] * 18))
    assert results == [
        (1, "doc-b", 2.0),
        (2, "doc-a", 1.0),
        (3, "doc-c", 0.0),
    ]


def test_query_defaults_to_50_results(tmp_path):
    doc_ids = [f"doc-{number}" for number in range(51)]
    vectors = np.eye(51, 20, dtype=np.float32)

    index = index_of(tmp_path, doc_ids, vectors)

    results = index.query(np.ones(20, dtype=np.float32))

    assert len(results) == 50


def test_query_scores_with_float32_query(tmp_path):
    index = index_of(
        tmp_path,
        ["doc-a"],
        np.array([[1, 2] + [0] * 18], dtype=np.float32),
    )

    with patch("exps.naive.np.argsort", wraps=np.argsort) as argsort:
        index.query(np.array([1, 1] + [0] * 18, dtype=np.float32))

    assert argsort.call_args.args[0].dtype == np.float32
