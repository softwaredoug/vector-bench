import csv
import json
import sys

from vector_bench.datasets import get_dataset
from vector_bench.main import main


def test_cli_writes_index_and_caches_ground_truth(tmp_path, monkeypatch):
    output_path = tmp_path / "embeddings.csv"
    cache_path = tmp_path / "cache"
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(cache_path))

    main(
        [
            "--dataset",
            "dougs_blog_data",
            "--top-k",
            "5",
            "--embeddings-file",
            str(output_path),
            "--",
            sys.executable,
            "-m",
            "vector_bench.naive_search",
        ]
    )

    corpus, _ = get_dataset("dougs_blog_data")
    with output_path.open(newline="") as output_file:
        rows = list(csv.reader(output_file))

    assert len(rows) == len(corpus)
    assert rows[0][0] == str(corpus.iloc[0]["doc_id"])
    assert len(rows[0]) > 1

    ground_truth_files = list(cache_path.glob("ground_truth_*.json"))
    assert len(ground_truth_files) == 1
    with ground_truth_files[0].open() as ground_truth_file:
        cached = json.load(ground_truth_file)
    assert cached["metadata"]["top_k"] == 5
    assert cached["rankings"]
    assert all(len(doc_ids) == 5 for doc_ids in cached["rankings"].values())
