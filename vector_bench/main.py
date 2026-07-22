"""Command-line entry point for benchmarking vector search applications."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .runner import launch_student
from .search import search
from .storage import read_queries


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
    """Load query vectors and ranked document IDs from prepared HDF5."""
    query_ids, query_vectors, ground_truth = read_queries(queries_path)
    if not query_ids:
        raise ValueError("Queries file must contain at least one query")
    return query_ids, np.asarray(query_vectors), ground_truth


if __name__ == "__main__":
    main()
