import csv
import json

from vector_bench.datasets import get_dataset
from vector_bench.prepare import main


def test_cli_writes_index_and_caches_ground_truth(tmp_path, monkeypatch):
    index_path = tmp_path / "embeddings.csv"
    queries_path = tmp_path / "queries.csv"
    cache_path = tmp_path / "cache"
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(cache_path))

    main(
        [
            "--dataset",
            "dougs_blog_data",
            "--top-k",
            "5",
            "--index-out",
            str(index_path),
            "--queries-out",
            str(queries_path),
        ]
    )

    corpus, _ = get_dataset("dougs_blog_data")
    with index_path.open(newline="") as output_file:
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
