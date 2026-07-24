import sys

from vector_bench.main import main as benchmark_main
from vector_bench.prepare import main as prepare_main


def test_recall_improves_when_naive_search_uses_more_dimensions(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(tmp_path / "cache"))

    index_path = tmp_path / "embeddings.h5"
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

    def run_with_dimensions(dimensions):
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
                "exps.naive",
                "--dimensions",
                str(dimensions),
            ]
        )
        output = capsys.readouterr().out.splitlines()
        return float(output[-1].split(",")[2])

    low_dimension_recall = run_with_dimensions(1)
    high_dimension_recall = run_with_dimensions(20)

    assert high_dimension_recall > low_dimension_recall
