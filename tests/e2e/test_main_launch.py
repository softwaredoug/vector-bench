import sys

from vector_bench.main import main as benchmark_main
from vector_bench.prepare import main as prepare_main


def test_main_launches_student_command_and_waits_for_ready(tmp_path):
    index_path = tmp_path / "embeddings.h5"
    queries_path = tmp_path / "queries.h5"
    prepare_main(
        [
            "--dataset",
            "dougs_blog_data",
            "--index-out",
            str(index_path),
            "--queries-out",
            str(queries_path),
        ]
    )
    benchmark_main(
        [
            "--index",
            str(index_path),
            "--queries",
            str(queries_path),
            "--",
            sys.executable,
            "-m",
            "exps.naive_search",
        ]
    )
