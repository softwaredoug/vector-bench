"""Portable HDF5 serialization for prepared vector-bench artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, cast

import h5py
import numpy as np


def _strings(values: Iterable[object]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=h5py.string_dtype())


def write_index(path: Path, doc_ids, vectors: np.ndarray) -> None:
    """Write document IDs and vectors without converting the vector dtype."""
    doc_ids = list(doc_ids)
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or len(doc_ids) != len(vectors):
        raise ValueError("Document IDs and vectors must have the same number of rows")

    with h5py.File(path, "w") as index_file:
        index_file.create_dataset("doc_ids", data=_strings(doc_ids))
        index_file.create_dataset("vectors", data=vectors)


def write_queries(
    path: Path,
    query_ids,
    query_vectors: np.ndarray,
    rankings: dict[str, list[str]],
) -> None:
    """Write query vectors and ranked document IDs in query/rank order."""
    query_ids = [str(query_id) for query_id in query_ids]
    query_vectors = np.asarray(query_vectors)
    if query_vectors.ndim != 2 or len(query_ids) != len(query_vectors):
        raise ValueError("Query IDs and vectors must have the same number of rows")

    max_rank = max((len(rankings[query_id]) for query_id in query_ids), default=0)
    ranked_ids = np.full((len(query_ids), max_rank), "", dtype=object)
    for query_index, query_id in enumerate(query_ids):
        ranked_ids[query_index, : len(rankings[query_id])] = rankings[query_id]

    with h5py.File(path, "w") as queries_file:
        queries_file.create_dataset("query_ids", data=_strings(query_ids))
        queries_file.create_dataset("vectors", data=query_vectors)
        queries_file.create_dataset(
            "ground_truth", data=ranked_ids, dtype=h5py.string_dtype()
        )


def read_index(path: Path) -> tuple[list[str], np.ndarray]:
    """Read document IDs and vectors from an HDF5 index."""
    with h5py.File(path, "r") as index_file:
        doc_id_dataset = cast(Any, index_file["doc_ids"])
        vector_dataset = cast(Any, index_file["vectors"])
        doc_ids = [_decode(value) for value in doc_id_dataset[:]]
        vectors = np.asarray(vector_dataset[:])
    return doc_ids, vectors


def read_queries(path: Path) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    """Read query vectors and ranked document IDs from HDF5."""
    with h5py.File(path, "r") as queries_file:
        query_id_dataset = cast(Any, queries_file["query_ids"])
        vector_dataset = cast(Any, queries_file["vectors"])
        ranked_id_dataset = cast(Any, queries_file["ground_truth"])
        query_ids = [_decode(value) for value in query_id_dataset[:]]
        vectors = np.asarray(vector_dataset[:])
        ranked_ids = ranked_id_dataset[:]

    ground_truth = {
        query_id: [_decode(value) for value in ranked_ids[index] if _decode(value)]
        for index, query_id in enumerate(query_ids)
    }
    return query_ids, vectors, ground_truth


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
