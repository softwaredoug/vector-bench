import sys

from vector_bench.main import main


def test_main_searches_student_and_prints_recall_and_latency(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("VECTOR_BENCH_DATA_DIR", str(tmp_path / "cache"))

    main(
        [
            "--dataset",
            "dougs_blog_data",
            "--top-k",
            "5",
            "--embeddings-file",
            str(tmp_path / "embeddings.csv"),
            "--",
            sys.executable,
            "-m",
            "vector_bench.naive_search",
        ]
    )

    output = capsys.readouterr().out.splitlines()

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
