"""Command-line entry point for benchmarking vector search applications."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .runner import launch_student
from .search import search


def main(argv: Sequence[str] | None = None) -> None:
    """Replay prepared queries against a student search application."""
    parser = argparse.ArgumentParser(prog="vector-bench")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("student_command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.student_command and args.student_command[0] == "--":
        args.student_command = args.student_command[1:]
    if not args.student_command:
        parser.error("a student command is required after --")

    query_ids, query_vectors, ground_truth = load_queries(args.queries)
    with launch_student(args.student_command, args.index) as student:
        search(
            student,
            query_ids,
            query_vectors,
            ground_truth,
            top_k=args.top_k,
        )


def load_queries(
    queries_path: Path,
) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    """Load query vectors and ranked document IDs from a prepared CSV."""
    query_ids = []
    query_vectors = []
    ground_truth = {}
    dimensions = None

    with queries_path.open(newline="") as queries_file:
        rows = csv.reader(queries_file)
        for row_number, row in enumerate(rows, start=1):
            if len(row) < 4:
                raise ValueError(
                    f"Query row {row_number} must contain query_id, doc_id, "
                    "rank, and a vector"
                )
            query_id, doc_id = row[0], row[1]
            try:
                rank = int(row[2])
                vector = np.asarray([float(value) for value in row[3:]], dtype=np.float32)
            except ValueError as error:
                raise ValueError(f"Invalid query row {row_number}") from error

            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ValueError("All query vectors must have the same dimensions")

            if rank == -1 and doc_id == "-1":
                if query_id in ground_truth:
                    raise ValueError(f"Duplicate query embedding for {query_id!r}")
                query_ids.append(query_id)
                query_vectors.append(vector)
                ground_truth[query_id] = []
            elif rank >= 1:
                if query_id not in ground_truth:
                    raise ValueError(
                        f"Query embedding must precede ranked rows for {query_id!r}"
                    )
                ground_truth[query_id].append((rank, doc_id))
            else:
                raise ValueError(
                    f"Query row {row_number} must use rank/doc_id -1 for a query "
                    "embedding or a positive rank for a document"
                )

    if not query_ids:
        raise ValueError("Queries file must contain at least one query")

    ranked_ground_truth = {
        query_id: [
            doc_id for _rank, doc_id in sorted(ranked_rows, key=lambda item: item[0])
        ]
        for query_id, ranked_rows in ground_truth.items()
    }
    return query_ids, np.asarray(query_vectors, dtype=np.float32), ranked_ground_truth


if __name__ == "__main__":
    main()
