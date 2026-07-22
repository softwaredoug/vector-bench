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
        doc_ids: list[str],
        doc_vectors: np.ndarray,
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> "VectorIndex":
        """Build an index from original document vectors.

        Students should customize this static method when replacing the
        baseline index. The document IDs must remain aligned with the rows in
        the stored vector array.
        """
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        if len(doc_ids) != len(doc_vectors):
            raise ValueError("Document IDs and vectors must have the same length")
        if doc_vectors.ndim != 2 or doc_vectors.shape[1] < dimensions:
            raise ValueError(
                f"Document vectors must have at least {dimensions} dimensions"
            )

        # Keeping only a prefix makes this intentionally naive implementation
        # small and gives students a clear place to try a better index later.
        return VectorIndex(
            doc_ids,
            np.asarray(doc_vectors[:, :dimensions], dtype=np.float32),
            dimensions,
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
    """Read the embeddings CSV and build the student index."""
    doc_ids = []
    vectors = []
    with index_path.open(newline="") as index_file:
        for row_number, row in enumerate(csv.reader(index_file), start=1):
            if len(row) < dimensions + 1:
                raise ValueError(
                    f"Index row {row_number} must contain a doc_id and "
                    f"at least {dimensions} dimensions"
                )
            doc_ids.append(row[0])
            vectors.append([float(value) for value in row[1:]])

    if not vectors:
        raise ValueError("Index must contain at least one document")
    return VectorIndex.index(
        doc_ids, np.asarray(vectors, dtype=np.float32), dimensions=dimensions
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
                    dtype=np.float32,
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

        def log_message(self, _format, *_args):
            """Keep request logs off stdout so READY remains unambiguous."""

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
