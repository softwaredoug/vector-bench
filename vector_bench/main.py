"""Command-line entry point for vector-bench."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

from .datasets import get_dataset
from .embeddings import DEFAULT_MODEL_NAME, embed_corpus, query_embeddings
from .runner import launch_student
from .search import search


def main(argv: Sequence[str] | None = None) -> None:
    """Load the selected dataset and create its corpus embeddings."""
    parser = argparse.ArgumentParser(prog="vector-bench")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--embeddings-file", type=Path)
    parser.add_argument("student_command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.student_command and args.student_command[0] == "--":
        args.student_command = args.student_command[1:]
    if not args.student_command:
        parser.error("a student command is required after --")
    corpus, judgments = get_dataset(args.dataset)
    ground_truth, embedding_lines = embed_corpus(
        corpus,
        judgments,
        dataset_name=args.dataset,
        model_name=args.model,
        top_k=args.top_k,
    )

    if args.embeddings_file is not None:
        embeddings_path = args.embeddings_file
        output_file = embeddings_path.open("w")
    else:
        output_file = NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        embeddings_path = Path(output_file.name)
    with output_file:
        output_file.writelines(embedding_lines)

    query_ids, query_vectors = query_embeddings(
        judgments,
        dataset_name=args.dataset,
        model_name=args.model,
    )
    with launch_student(args.student_command, embeddings_path) as student:
        search(
            student,
            query_ids,
            query_vectors,
            ground_truth,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
