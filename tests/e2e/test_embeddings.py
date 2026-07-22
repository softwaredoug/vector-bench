import json
import h5py

from vector_bench.datasets import get_dataset
from vector_bench.prepare import main


def test_cli_writes_index_and_caches_ground_truth(tmp_path, monkeypatch):
    index_path = tmp_path / "embeddings.h5"
    queries_path = tmp_path / "queries.h5"
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
    with h5py.File(index_path) as output_file:
        assert len(output_file["doc_ids"]) == len(corpus)
        first_doc_id = output_file["doc_ids"][0]
        if isinstance(first_doc_id, bytes):
            first_doc_id = first_doc_id.decode()
        assert first_doc_id == str(corpus.iloc[0]["doc_id"])
        assert output_file["vectors"].shape[1] > 1

    ground_truth_files = list(cache_path.glob("ground_truth_*.json"))
    assert len(ground_truth_files) == 1
    with ground_truth_files[0].open() as ground_truth_file:
        cached = json.load(ground_truth_file)
    assert cached["metadata"]["top_k"] == 5
    assert cached["rankings"]
    assert all(len(doc_ids) == 5 for doc_ids in cached["rankings"].values())
