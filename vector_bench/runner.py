"""Launch and supervise a student vector search process."""

from __future__ import annotations

import os
import selectors
import socket
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def _find_free_port() -> int:
    """Ask the operating system for an available local TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class StudentProcess:
    """A student process that has completed indexing and is ready to query."""

    process: subprocess.Popen[str]
    port: int

    def stop(self) -> None:
        """Terminate the student process and avoid leaving a child behind."""
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def __enter__(self) -> "StudentProcess":
        return self

    def __exit__(self, *_args) -> None:
        self.stop()


def launch_student(
    command: Sequence[str],
    index_path: Path,
    port: int | None = None,
    ready_timeout: float = 3000,
) -> StudentProcess:
    """Launch a student command and wait until it prints ``READY``."""
    if not command:
        raise ValueError("A student command is required after --")

    selected_port = port or _find_free_port()
    full_command = [
        *command,
        "--index",
        str(index_path),
        "--port",
        str(selected_port),
    ]
    process = subprocess.Popen(
        full_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        _wait_for_ready(process, ready_timeout)
        _start_output_drainer(process.stdout)
        _start_output_drainer(process.stderr)
    except Exception:
        _stop_process(process)
        raise
    return StudentProcess(process, selected_port)


def _wait_for_ready(process: subprocess.Popen[str], timeout: float) -> None:
    """Read child output until READY or raise a useful startup error."""
    if process.stdout is None:
        raise RuntimeError("Student process stdout is unavailable")

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    pending = b""
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Student process did not print READY in time")
            if not selector.select(remaining):
                raise TimeoutError("Student process did not print READY in time")
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                error = process.stderr.read() if process.stderr else ""
                raise RuntimeError(
                    "Student process exited before READY"
                    + (f": {error.strip()}" if error.strip() else "")
                )
            pending += chunk
            while b"\n" in pending:
                raw_line, pending = pending.split(b"\n", 1)
                line = raw_line.decode()
                print(f"Student process output: {line.strip()}")
                if line.strip() == "READY":
                    print("Student process is ready")
                    return
    finally:
        selector.close()


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Stop a process during failed startup."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _start_output_drainer(stream) -> None:
    """Consume child output after startup so its pipes cannot fill."""
    if stream is None:
        return
    thread = threading.Thread(target=_drain_output, args=(stream,), daemon=True)
    thread.start()


def _drain_output(stream) -> None:
    """Read and discard output from a running child process."""
    while True:
        try:
            if not os.read(stream.fileno(), 4096):
                return
        except OSError:
            return
