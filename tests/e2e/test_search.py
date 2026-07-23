import sys

from vector_bench.main import main as benchmark_main
from vector_bench.prepare import main as prepare_main


def test_main_searches_student_and_prints_recall_and_latency(
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
    output = output[-11:]

    assert len(output) == 11
    average = output[-1].split(",")
    first_result = output[0].split(",")
    assert average[0] == ""
    assert len(average) == 3
    assert float(average[1]) >= 0
    assert 0 <= float(average[2]) <= 1
    assert len(first_result) == 3
    assert float(first_result[1]) >= 0
    assert 0 <= float(first_result[2]) <= 1
