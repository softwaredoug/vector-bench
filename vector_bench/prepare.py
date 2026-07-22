"""Prepare corpus and query embedding files for vector-bench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .datasets import get_dataset
from .embeddings import (
    DEFAULT_MODEL_NAME,
    corpus_embedding_artifacts,
    query_embeddings,
)
from .storage import write_index, write_queries


def main(argv: Sequence[str] | None = None) -> None:
    """Create the HDF5 index and query files for a benchmark run."""
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

    write_index(args.index_out, corpus["doc_id"], corpus_vectors)

    query_ids = [str(query_id) for query_id in query_ids]
    query_rankings = {}
    for query_id, query_vector in zip(query_ids, query_vectors):
        query_rankings[str(query_id)] = rankings[str(query_id)]

    sort_order = sorted(range(len(query_ids)), key=lambda index: query_ids[index])
    sorted_query_ids = [query_ids[index] for index in sort_order]
    sorted_query_vectors = query_vectors[sort_order]
    write_queries(args.queries_out, sorted_query_ids, sorted_query_vectors, query_rankings)


if __name__ == "__main__":
    main()
