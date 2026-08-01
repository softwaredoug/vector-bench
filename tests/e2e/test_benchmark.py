import sys

from vector_bench.main import main as benchmark_main
from vector_bench.prepare import main as prepare_main
from vector_bench.runner import launch_student


def test_launch_student_handles_buffered_startup_output(tmp_path):
    with launch_student(
        [
            sys.executable,
            "-c",
            "import time; print('INDEX LOADED'); print('READY', flush=True); time.sleep(10)",
        ],
        tmp_path / "index.h5",
        ready_timeout=1,
    ):
        pass


def test_launch_student_drains_output_after_ready(tmp_path):
    with launch_student(
        [
            sys.executable,
            "-c",
            (
                "import sys, time; "
                "print('READY', flush=True); "
                "sys.stderr.write('x' * 200000); sys.stderr.flush(); "
                "print('DONE', flush=True)"
            ),
        ],
        tmp_path / "index.h5",
        ready_timeout=1,
    ) as student:
        student.process.wait(timeout=1)
        assert student.process.returncode == 0


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
            "--concurrency",
            "2",
            "--",
            sys.executable,
            "-m",
            "exps.naive",
        ]
    )

    captured = capsys.readouterr()
    output = captured.out.splitlines()
    metrics = output[-11:]
    assert len(metrics) == 11
    assert metrics[-1].startswith(",")
    assert all(len(row.split(",")) == 3 for row in metrics)
    query_logs = [line for line in captured.err.splitlines() if "Query " in line]
    assert len(query_logs) == 10
    assert all("qps=" in line for line in query_logs)
