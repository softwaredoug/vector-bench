"""Shared HTTP and HDF5 plumbing for standalone search applications."""

import argparse
import csv
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs

import h5py
import numpy as np


class Index(Protocol):
    """Interface required by the standalone search server."""

    dimensions: int

    @staticmethod
    def index(
        doc_ids: h5py.Dataset,
        vectors: h5py.Dataset,
        dimensions: int,
    ) -> "Index":
        ...

    def query(
        self, query_vector: np.ndarray, top_k: int | None = None
    ) -> list[tuple[int, str]]:
        ...


@contextmanager
def datasets(path: Path) -> Generator[tuple[h5py.Dataset, h5py.Dataset], None, None]:
    """Open the document IDs and vectors in an HDF5 index."""
    with h5py.File(path, "r") as index_file:
        doc_id_dataset = index_file["doc_ids"]
        vector_dataset = index_file["vectors"]
        if not isinstance(doc_id_dataset, h5py.Dataset) or not isinstance(
            vector_dataset, h5py.Dataset
        ):
            raise ValueError("Expected 'doc_ids' and 'vectors' datasets in HDF5 file")
        yield doc_id_dataset, vector_dataset


def load_index(
    index_type: type[Index], index_path: Path, dimensions: int
) -> Index:
    """Load HDF5 data and build an index using the supplied class."""
    with datasets(index_path) as (doc_ids, vectors):
        return index_type.index(doc_ids, vectors, dimensions=dimensions)


def make_query_handler(index: Index):
    """Create an HTTP handler with access to the in-memory document index."""

    class QueryHandler(BaseHTTPRequestHandler):
        """Handle the one POST endpoint required by the benchmark."""

        def do_POST(self):  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path != "/query":
                self.send_error(404, "Only POST /query is supported")
                return

            try:
                request_length = int(self.headers["Content-Length"])
                request_body = self.rfile.read(request_length).decode()
                fields = parse_qs(request_body)
                query_id = fields["query_id"][0]
                query_vector = np.asarray(
                    [float(value) for value in fields["vector"][0].split(",")],
                    dtype=np.float64,
                )
                if len(query_vector) < index.dimensions:
                    raise ValueError(
                        "Query vector must contain at least "
                        f"{index.dimensions} dimensions"
                    )
            except (KeyError, TypeError, ValueError) as error:
                self.send_error(400, str(error))
                return

            output = StringIO()
            writer = csv.writer(output, lineterminator="\n")
            for rank, doc_id in index.query(query_vector):
                writer.writerow([rank, query_id, doc_id])

            response = output.getvalue().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    return QueryHandler


def serve(index_type: type[Index], argv: Sequence[str] | None = None) -> None:
    """Load an index class, start its HTTP server, and serve query requests."""
    parser = argparse.ArgumentParser(prog="naive-vector-search")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dimensions", type=int, default=60)
    args = parser.parse_args(argv)

    index = load_index(index_type, args.index, dimensions=args.dimensions)
    print("Inedx loaded")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_query_handler(index))

    try:
        print("READY", flush=True)
        server.serve_forever()
    finally:
        server.server_close()
