"""Command-line entry point for vector-bench."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

from .datasets import get_dataset
from .embeddings import DEFAULT_MODEL_NAME, embed_corpus


def main(argv: Sequence[str] | None = None) -> None:
    """Load the selected dataset and create its corpus embeddings."""
    parser = argparse.ArgumentParser(prog="vector-bench")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--embeddings-file", type=Path)
    args = parser.parse_args(argv)
    corpus, _judgments = get_dataset(args.dataset)
    embedding_lines = embed_corpus(corpus, model_name=args.model)

    if args.embeddings_file is not None:
        output_file = args.embeddings_file.open("w")
    else:
        output_file = NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    with output_file:
        output_file.writelines(embedding_lines)


if __name__ == "__main__":
    main()
