import csv

from vector_bench.datasets import get_dataset
from vector_bench.main import main


def test_cli_writes_dougs_blog_data_embeddings_for_student_tool(tmp_path):
    output_path = tmp_path / "embeddings.csv"

    main(
        [
            "--dataset",
            "dougs_blog_data",
            "--embeddings-file",
            str(output_path),
        ]
    )

    corpus, _ = get_dataset("dougs_blog_data")
    with output_path.open(newline="") as output_file:
        rows = list(csv.reader(output_file))

    assert len(rows) == len(corpus)
    assert rows[0][0] == str(corpus.iloc[0]["doc_id"])
    assert len(rows[0]) > 1
