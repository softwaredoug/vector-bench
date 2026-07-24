import os
import signal
import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import h5py
import pytest

import numpy as np


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
    with h5py.File(index_path, "w") as index_file:
        index_file.create_dataset(
            "doc_ids", data=[doc_id.encode() for doc_id in ["doc-a", "doc-b", "doc-c"]]
        )
        index_file.create_dataset("vectors", data=vectors)
    return index_path


def test_naive_search_indexes_and_queries_embeddings(index_path):
    port = 18765
    print("Launching process")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from exps import serve; from exps.naive import VectorIndex; serve.serve(VectorIndex)",
            "--index",
            str(index_path),
            "--port",
            str(port),
            "--dimensions",
            "20",
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
                {
                    "query_id": "q1",
                    "vector": ",".join(["1", "1"] + ["0"] * 20),
                    "top_k": "2",
                }
            ).encode(),
            method="POST",
        )
        with urlopen(request) as response:
            result = response.read().decode().splitlines()

        assert result == [
            "1,q1,doc-b",
            "2,q1,doc-a",
        ]
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_naive_search_test_mode_queries_corpus_until_interrupted(index_path):
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from exps.naive import main; main()",
            "--index",
            str(index_path),
            "--dimensions",
            "2",
            "--test-index-size",
            "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        assert process.stdout is not None
        output = []
        while not any("TEST MODE" in line for line in output):
            line = process.stdout.readline()
            assert line, process.stderr.read() if process.stderr else ""
            output.append(line)
        while not any("doc-" in line for line in output):
            line = process.stdout.readline()
            assert line, process.stderr.read() if process.stderr else ""
            output.append(line)
        os.kill(process.pid, signal.SIGINT)
        output.extend(process.communicate(timeout=5)[0].splitlines(keepends=True))

        assert process.returncode == 0
        assert any("doc-a" in line or "doc-b" in line for line in output)
        assert not any("doc-c" in line for line in output)
        assert any("query_doc_id=doc-" in line for line in output)
        assert any(
            "results=[(" in line and line.count(",") >= 6 for line in output
        )
        assert any("0.000707106427633422" in line for line in output)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
