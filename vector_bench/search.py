"""Query a ready student application and calculate recall and latency."""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from time import perf_counter
from typing import TextIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from .runner import StudentProcess

MAX_TOP_K = 50


def search(
    student: StudentProcess,
    query_ids: Sequence[str],
    query_vectors: np.ndarray,
    ground_truth: dict[str, list[str]],
    top_k: int,
    output: TextIO | None = None,
    concurrency: int = 1,
) -> None:
    """Run all judged queries and print per-query and average metrics."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero")
    top_k = min(top_k, MAX_TOP_K)
    if len(query_ids) != len(query_vectors):
        raise ValueError("Query IDs and vectors must have the same length")

    results = []
    total_recall = 0
    total_latency = 0
    num_queries_run = 0
    benchmark_started = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _query,
                student.port,
                query_id,
                query_vector,
                ground_truth.get(str(query_id), [])[:top_k],
                top_k,
            )
            for query_id, query_vector in zip(query_ids, query_vectors)
        ]
        for future in futures:
            query_id, latency, recall = future.result()

            total_recall += recall
            total_latency += latency
            num_queries_run += 1
            avg_recall = total_recall / num_queries_run
            avg_latency = total_latency / num_queries_run
            elapsed = perf_counter() - benchmark_started
            qps = num_queries_run / elapsed if elapsed else 0.0

            print(
                f"{num_queries_run} -- Query {query_id}: "
                f"latency={latency:.6f}s ({avg_latency:.6f}s), "
                f"qps={qps:.2f}, recall={recall:.4f} ({avg_recall:.4f})",
                file=sys.stderr,
            )
            results.append((query_id, latency, recall))

    stream = output or sys.stdout
    for query_id, latency, recall in results:
        total_recall += recall
        total_latency += latency
        print(f"{query_id},{latency},{recall}", file=stream)

    average_latency = sum(result[1] for result in results) / len(results)
    average_recall = sum(result[2] for result in results) / len(results)
    print(f",{average_latency},{average_recall}", file=stream)


def _query(
    port: int,
    query_id: str,
    query_vector: np.ndarray,
    expected_doc_ids: Sequence[str],
    top_k: int,
) -> tuple[str, float, float]:
    """Run one query and calculate its latency and recall."""
    started = perf_counter()
    retrieved_doc_ids = _post_query(port, query_id, query_vector, top_k=top_k)
    latency = perf_counter() - started
    recall = _recall(retrieved_doc_ids[:top_k], expected_doc_ids)
    return query_id, latency, recall


def _post_query(
    port: int, query_id: str, query_vector: np.ndarray, top_k: int
) -> list[str]:
    """POST one form-encoded query and parse returned document IDs."""
    body = urlencode(
        {
            "query_id": query_id,
            "vector": ",".join(str(value) for value in query_vector),
            "top_k": str(top_k),
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
