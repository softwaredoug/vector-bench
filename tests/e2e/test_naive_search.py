import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pytest

import numpy as np

from vector_bench.storage import write_index


@pytest.fixture
def index_path(tmp_path):
    index_path = tmp_path / "embeddings.h5"
    vectors = np.array(
        [
            [1, 0, *([0] * 15), 1000, 0, 0],
            [1, 1, *([0] * 15), 0, 0, 0],
            [0, 0, *([0] * 15), 2000, 0, 0],
        ],
        dtype=np.float64,
    )
    write_index(
        index_path,
        ["doc-a", "doc-b", "doc-c"],
        vectors
    )
    return index_path


def test_naive_search_indexes_and_queries_embeddings(index_path):
    port = 18765
    print("Launching process")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from vector_bench.naive_search import main; main()",
            "--index",
            str(index_path),
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print("Process launched")

    try:
        # assert_no_stderr(process)
        assert process.stdout is not None
        assert any(
            line.strip() == "READY"
            for line in iter(process.stdout.readline, "")
        )

        request = Request(
            f"http://127.0.0.1:{port}/query",
            data=urlencode(
                {"query_id": "q1", "vector": ",".join(["1", "1"] + ["0"] * 20)}
            ).encode(),
            method="POST",
        )
        with urlopen(request) as response:
            result = response.read().decode().splitlines()

        assert result == [
            "1,q1,doc-b",
            "2,q1,doc-a",
            "3,q1,doc-c",
        ]
    finally:
        process.terminate()
        process.wait(timeout=5)
