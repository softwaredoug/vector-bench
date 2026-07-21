"""Command-line entry point for vector-bench."""

import argparse
from collections.abc import Sequence

from .datasets import get_dataset
from .embeddings import DEFAULT_MODEL_NAME, embed_corpus


def main(argv: Sequence[str] | None = None) -> tuple[object, object, object, object]:
    """Load the selected dataset and create its corpus embeddings."""
    parser = argparse.ArgumentParser(prog="vector-bench")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args(argv)
    corpus, judgments = get_dataset(args.dataset)
    embeddings, model = embed_corpus(corpus, model_name=args.model)
    return corpus, judgments, embeddings, model


if __name__ == "__main__":
    main()
