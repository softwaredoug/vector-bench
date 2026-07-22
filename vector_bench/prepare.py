"""Prepare corpus and query embedding files for vector-bench."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from .datasets import get_dataset
from .embeddings import (
    DEFAULT_MODEL_NAME,
    corpus_embedding_artifacts,
    embedding_csv_lines,
    query_embeddings,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Create the index and query CSV files for a benchmark run."""
    parser = argparse.ArgumentParser(prog="prepare")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--num-queries", type=int)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--queries-out", type=Path, required=True)
    args = parser.parse_args(argv)

    corpus, judgments = get_dataset(args.dataset)
    if args.num_queries is not None:
        if args.num_queries <= 0:
            parser.error("--num-queries must be greater than zero")
        query_ids = judgments["query_id"].drop_duplicates().head(args.num_queries)
        judgments = judgments[judgments["query_id"].isin(query_ids)]

    corpus_vectors, rankings = corpus_embedding_artifacts(
        corpus,
        judgments,
        dataset_name=args.dataset,
        model_name=args.model,
        top_k=args.top_k,
    )
    query_ids, query_vectors = query_embeddings(
        judgments,
        dataset_name=args.dataset,
        model_name=args.model,
    )

    with args.index_out.open("w", newline="") as index_file:
        index_file.writelines(embedding_csv_lines(corpus, corpus_vectors))

    vectors_by_doc_id = {
        str(doc_id): vector
        for doc_id, vector in zip(corpus["doc_id"], corpus_vectors)
    }
    query_rows = []
    for query_id, query_vector in zip(query_ids, query_vectors):
        query_rows.append((str(query_id), "-1", -1, query_vector))
        for rank, doc_id in enumerate(rankings[str(query_id)], start=1):
            query_rows.append(
                (str(query_id), str(doc_id), rank, vectors_by_doc_id[str(doc_id)])
            )

    query_rows.sort(key=lambda row: (row[0], row[2]))
    with args.queries_out.open("w", newline="") as queries_file:
        writer = csv.writer(queries_file, lineterminator="\n")
        for query_id, doc_id, rank, vector in query_rows:
            writer.writerow([query_id, doc_id, rank, *vector])


if __name__ == "__main__":
    main()
