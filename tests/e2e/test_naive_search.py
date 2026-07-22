import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from vector_bench.storage import write_index


def test_naive_search_indexes_and_queries_embeddings(tmp_path):
    index_path = tmp_path / "embeddings.h5"
    write_index(
        index_path,
        ["doc-a", "doc-b", "doc-c"],
        np.array(
            [
                [1, 0, *([0] * 19), 1000, 0],
                [0, 2, *([0] * 19), 0, 0],
                [0, 0, *([0] * 19), 2000, 0],
            ],
            dtype=np.float64,
        ),
    )

    port = 18765
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

    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"

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
