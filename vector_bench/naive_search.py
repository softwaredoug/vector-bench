"""A deliberately simple brute-force vector search application.

This module is also intended to be readable as a student starter application.
It is a separate process from the benchmark and communicates with the
benchmark only through the index file and HTTP protocol documented in the
project README.
"""

import argparse
import csv
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
import h5py

from .storage import datasets


# The benchmark provides full-size embeddings, but this demo intentionally
# uses only a small prefix so the search implementation stays straightforward.
DEFAULT_DIMENSIONS = 20


@dataclass
class VectorIndex:
    """The in-memory document IDs and vectors used by the search server."""

    doc_ids: list[str]
    doc_vectors: np.ndarray
    dimensions: int

    @staticmethod
    def index(
        doc_ids: h5py.Dataset,
        vectors: h5py.Dataset,
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> "VectorIndex":
        """Build an index from original document vectors.

        Students should customize this static method when replacing the
        baseline index. The document IDs must remain aligned with the rows in
        the stored vector array.
        """
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        rows, orig_dims = vectors.shape

        if orig_dims < dimensions:
            raise ValueError(
                f"vectors must contain at least {dimensions} dimensions"
            )

        # Pre-allocate lower dimensional
        index = np.empty((
            rows,
            dimensions,
        ), dtype=np.float64)

        index_doc_ids = []

        # Concat vectors + doc_ids
        for doc_id, vector in zip(doc_ids, vectors):
            index_doc_ids.append(doc_id.decode())
            index[len(index_doc_ids) - 1] = vector[:dimensions]

        # Keeping only a prefix makes this intentionally naive implementation
        # small and gives students a clear place to try a better index later.
        return VectorIndex(
            index_doc_ids,
            doc_vectors=index,
            dimensions=dimensions,
        )

    def query(self, query_vector: np.ndarray, top_k: int | None = None):
        """Return ranked document IDs for one query vector.

        Students should customize this method when replacing the baseline
        search. The baseline uses a brute-force dot product against every
        indexed document. ``rank`` starts at one for the HTTP response.
        """
        if len(query_vector) < self.dimensions:
            raise ValueError(
                f"Query vector must contain at least {self.dimensions} dimensions"
            )

        # Both arrays have already been limited to 20 dimensions. A dot
        # product gives one similarity score for every indexed document.
        scores = self.doc_vectors @ query_vector[: self.dimensions]
        ranked_indexes = np.argsort(-scores, kind="stable")
        if top_k is not None:
            ranked_indexes = ranked_indexes[:top_k]
        return [
            (rank, self.doc_ids[int(document_index)])
            for rank, document_index in enumerate(ranked_indexes, start=1)
        ]


def load_index(
    index_path: Path, dimensions: int = DEFAULT_DIMENSIONS
) -> VectorIndex:
    """Read the HDF5 embeddings and build the student index."""
    with datasets(index_path) as (doc_ids, vectors):
        return VectorIndex.index(
            doc_ids,
            vectors,
            dimensions=dimensions
        )


def make_query_handler(vector_index: VectorIndex):
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
                if len(query_vector) < vector_index.dimensions:
                    raise ValueError(
                        "Query vector must contain at least "
                        f"{vector_index.dimensions} dimensions"
                    )
            except (KeyError, TypeError, ValueError) as error:
                self.send_error(400, str(error))
                return

            output = StringIO()
            writer = csv.writer(output, lineterminator="\n")
            for rank, doc_id in vector_index.query(query_vector):
                writer.writerow([rank, query_id, doc_id])

            response = output.getvalue().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    return QueryHandler


def main(argv=None) -> None:
    """Load the index, start the server, and wait for query requests."""
    parser = argparse.ArgumentParser(prog="naive-vector-search")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    args = parser.parse_args(argv)

    vector_index = load_index(args.index, dimensions=args.dimensions)
    handler = make_query_handler(vector_index)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)

    try:
        # The benchmark waits for this exact line before sending queries.
        print("READY", flush=True)
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
