"""Create embeddings for vector-bench corpora."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from cheat_at_search.embeddings import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    load_model,
    load_or_create_embeddings,
)
from tqdm.auto import tqdm


def passage_fn(row: Any) -> str:
    """Build the text used to embed one corpus row."""
    title = row.get("title")
    description = row.get("description", "")

    if title:
        return f"{title}\n\n{description}"
    return description


def embed_corpus(
    corpus: Any,
    judgments: Any,
    dataset_name: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    top_k: int = 1000,
    show_progress: bool = True,
) -> tuple[dict[str, list[str]], np.ndarray]:
    """Create embeddings and ground truth rankings."""
    embeddings, rankings = corpus_embedding_artifacts(
        corpus,
        judgments,
        dataset_name=dataset_name,
        model_name=model_name,
        device=device,
        chunk_size=chunk_size,
        top_k=top_k,
        show_progress=show_progress,
    )
    return rankings, embeddings


def corpus_embedding_artifacts(
    corpus: Any,
    judgments: Any,
    dataset_name: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    top_k: int = 1000,
    show_progress: bool = True,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Create corpus embeddings and their ranked query ground truth."""
    embeddings, _model = load_or_create_embeddings(
        corpus,
        passage_fn=passage_fn,
        model_name=model_name,
        device=device,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )
    rankings = ground_truth(
        corpus,
        judgments,
        embeddings,
        dataset_name=dataset_name,
        model_name=model_name,
        device=device,
        top_k=top_k,
        show_progress=show_progress,
    )
    return np.asarray(embeddings), rankings


def ground_truth(
    corpus: Any,
    judgments: Any,
    corpus_embeddings: np.ndarray,
    dataset_name: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    top_k: int = 1000,
    show_progress: bool = True,
) -> dict[str, list[str]]:
    """Return or create dot-product rankings for judged queries."""
    print("Loading ground truth")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if len(corpus) != len(corpus_embeddings):
        raise ValueError("Corpus and embeddings must contain the same number of rows")

    queries = judgments[["query_id", "query"]].drop_duplicates("query_id")
    signature = _ground_truth_signature(
        corpus,
        queries,
        dataset_name,
        model_name,
        top_k,
        corpus_embeddings.shape[1],
    )
    print(f"Ground truth signature: {signature}")
    cache_path = _cache_dir() / f"ground_truth_{signature}.json"
    if cache_path.exists():
        print("Opening cached ground truth")
        with cache_path.open(encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
        return cached["rankings"]

    query_embeddings = _query_embeddings(
        queries,
        dataset_name=dataset_name,
        model_name=model_name,
        device=device,
        show_progress=show_progress,
    )
    doc_ids = [str(doc_id) for doc_id in corpus["doc_id"]]
    rankings = {}
    query_iterator = tqdm(
        zip(queries["query_id"], query_embeddings),
        total=len(queries),
        desc="Ranking queries",
        disable=not show_progress,
    )
    for query_id, query_embedding in query_iterator:
        scores = corpus_embeddings @ query_embedding
        ranked_indexes = np.argsort(-scores, kind="stable")[:top_k]
        rankings[str(query_id)] = [doc_ids[index] for index in ranked_indexes]
    _write_ground_truth(cache_path, dataset_name, model_name, top_k, rankings)
    return rankings


def query_embeddings(
    judgments: Any,
    dataset_name: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    show_progress: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Return unique judged query IDs and their cached embedding vectors."""
    queries = judgments[["query_id", "query"]].drop_duplicates("query_id")
    vectors = _query_embeddings(
        queries,
        dataset_name=dataset_name,
        model_name=model_name,
        device=device,
        show_progress=show_progress,
    )
    return [str(query_id) for query_id in queries["query_id"]], vectors


def _cache_dir() -> Path:
    """Return the vector-bench cache directory, creating it when needed."""
    configured_path = os.environ.get("VECTOR_BENCH_DATA_DIR")
    cache_path = (
        Path(configured_path)
        if configured_path
        else Path.home() / ".cache" / "vector-bench"
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _ground_truth_signature(
    corpus: Any,
    queries: Any,
    dataset_name: str,
    model_name: str,
    top_k: int,
    embedding_dimension: int,
) -> str:
    """Create a cache key from all inputs that affect the rankings."""
    payload = {
        "dataset": dataset_name,
        "model": model_name,
        "similarity": "dot_product",
        "top_k": top_k,
        "embedding_dimension": embedding_dimension,
        "doc_ids": [str(doc_id) for doc_id in corpus["doc_id"]],
        "queries": [
            [str(query_id), str(query)]
            for query_id, query in zip(queries["query_id"], queries["query"])
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _query_embeddings(
    queries: Any,
    dataset_name: str,
    model_name: str,
    device: str | None,
    show_progress: bool,
) -> np.ndarray:
    """Load cached query vectors or encode the unique judged queries."""
    signature = _query_signature(queries, dataset_name, model_name)
    cache_path = _cache_dir() / f"query_embeddings_{signature}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    model: Any = load_model(model_name, device=device)
    query_texts = [str(query) for query in queries["query"]]
    embeddings = model.encode(
        query_texts,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings)
    np.save(cache_path, embeddings)
    return embeddings


def _query_signature(queries: Any, dataset_name: str, model_name: str) -> str:
    """Create a cache key for query texts and their embedding model."""
    payload = {
        "dataset": dataset_name,
        "model": model_name,
        "queries": [
            [str(query_id), str(query)]
            for query_id, query in zip(queries["query_id"], queries["query"])
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_ground_truth(
    cache_path: Path,
    dataset_name: str,
    model_name: str,
    top_k: int,
    rankings: dict[str, list[str]],
) -> None:
    """Write rankings atomically so interrupted runs do not corrupt caches."""
    payload = {
        "metadata": {
            "dataset": dataset_name,
            "model": model_name,
            "top_k": top_k,
        },
        "rankings": rankings,
    }
    temporary_path = cache_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file)
    temporary_path.replace(cache_path)
