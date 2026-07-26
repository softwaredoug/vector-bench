import os
import signal
import subprocess
import sys

import h5py
import numpy as np


def test_turboquant_graphs_isotropy_before_and_after_rotation(tmp_path):
    index_path = tmp_path / "embeddings.h5"
    vectors = np.arange(24, dtype=np.float64).reshape(6, 4)
    with h5py.File(index_path, "w") as index_file:
        index_file.create_dataset(
            "doc_ids", data=[f"doc-{number}".encode() for number in range(len(vectors))]
        )
        index_file.create_dataset("vectors", data=vectors)

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from exps.turboquant import main; main()",
            "--index",
            str(index_path),
            "--dimensions",
            "4",
            "--test-index-size",
            "2",
            "--graph-isotropy",
        ],
        cwd=tmp_path,
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

        os.kill(process.pid, signal.SIGINT)
        process.communicate(timeout=5)

        assert process.returncode == 0
        for filename in (
            "graph_coord_before.png",
            "graph_coord_after.png",
            "graph_eigen_before.png",
            "graph_eigen_after.png",
        ):
            assert (tmp_path / filename).read_bytes().startswith(b"\x89PNG")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
