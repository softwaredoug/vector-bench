"""Query a ready student application and calculate recall and latency."""

from __future__ import annotations

import csv
import sys
from collections.abc import Sequence
from time import perf_counter
from typing import TextIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from .runner import StudentProcess


def search(
    student: StudentProcess,
    query_ids: Sequence[str],
    query_vectors: np.ndarray,
    ground_truth: dict[str, list[str]],
    top_k: int,
    output: TextIO | None = None,
) -> None:
    """Run all judged queries and print per-query and average metrics."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if len(query_ids) != len(query_vectors):
        raise ValueError("Query IDs and vectors must have the same length")

    results = []
    for query_id, query_vector in zip(query_ids, query_vectors):
        started = perf_counter()
        retrieved_doc_ids = _post_query(student.port, query_id, query_vector)
        latency = perf_counter() - started
        expected_doc_ids = ground_truth.get(str(query_id), [])[:top_k]
        retrieved_doc_ids = retrieved_doc_ids[:top_k]
        recall = _recall(retrieved_doc_ids, expected_doc_ids)
        results.append((query_id, latency, recall))

    stream = output or sys.stdout
    for query_id, latency, recall in results:
        print(f"{query_id},{latency},{recall}", file=stream)

    average_latency = sum(result[1] for result in results) / len(results)
    average_recall = sum(result[2] for result in results) / len(results)
    print(f",{average_latency},{average_recall}", file=stream)


def _post_query(port: int, query_id: str, query_vector: np.ndarray) -> list[str]:
    """POST one form-encoded query and parse returned document IDs."""
    body = urlencode(
        {
            "query_id": query_id,
            "vector": ",".join(str(value) for value in query_vector),
        }
    ).encode()
    request = Request(
        f"http://127.0.0.1:{port}/query",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request) as response:
        rows = csv.reader(response.read().decode().splitlines())
        return [row[2] for row in rows if len(row) >= 3]


def _recall(retrieved_doc_ids: Sequence[str], expected_doc_ids: Sequence[str]) -> float:
    """Calculate top-k recall as overlap divided by true top-k count."""
    if not expected_doc_ids:
        return 0.0
    return len(set(retrieved_doc_ids) & set(expected_doc_ids)) / len(expected_doc_ids)
