import sys

from vector_bench.main import main as benchmark_main
from vector_bench.prepare import main as prepare_main


def test_benchmark_main_replays_prepared_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(tmp_path / "cache"))
    index_path = tmp_path / "index.h5"
    queries_path = tmp_path / "queries.h5"

    prepare_main(
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
    capsys.readouterr()

    benchmark_main(
        [
            "--index",
            str(index_path),
            "--queries",
            str(queries_path),
            "--top-k",
            "5",
            "--",
            sys.executable,
            "-m",
            "vector_bench.naive_search",
        ]
    )

    output = capsys.readouterr().out.splitlines()
    assert len(output) == 11
    assert output[-1].startswith(",")
    assert all(len(row.split(",")) == 3 for row in output)
