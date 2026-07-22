import sys

from vector_bench.main import main


def test_main_launches_student_command_and_waits_for_ready(tmp_path):
    main(
        [
            "--dataset",
            "dougs_blog_data",
            "--embeddings-file",
            str(tmp_path / "embeddings.csv"),
            "--",
            sys.executable,
            "-m",
            "vector_bench.naive_search",
        ]
    )
