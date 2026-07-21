"""A deliberately simple brute-force vector search application.

This module is also intended to be readable as a student starter application.
It is a separate process from the benchmark and communicates with the
benchmark only through the index file and HTTP protocol documented in the
project README.
"""

import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np


# The benchmark provides full-size embeddings, but this demo intentionally
# uses only a small prefix so the search implementation stays straightforward.
NUM_DIMENSIONS = 20


def load_index(index_path: Path) -> tuple[list[str], np.ndarray]:
    """Read document IDs and the first 20 vector dimensions from a CSV file."""
    doc_ids = []
    vectors = []

    with index_path.open(newline="") as index_file:
        for row_number, row in enumerate(csv.reader(index_file), start=1):
            if len(row) < NUM_DIMENSIONS + 1:
                raise ValueError(
                    f"Index row {row_number} must contain a doc_id and "
                    f"at least {NUM_DIMENSIONS} dimensions"
                )
            doc_ids.append(row[0])
            vectors.append([float(value) for value in row[1 : NUM_DIMENSIONS + 1]])

    if not vectors:
        raise ValueError("Index must contain at least one document")

    return doc_ids, np.asarray(vectors, dtype=np.float32)


def make_query_handler(doc_ids: list[str], document_vectors: np.ndarray):
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
                if len(query_vector) < NUM_DIMENSIONS:
                    raise ValueError(
                        f"Query vector must contain at least {NUM_DIMENSIONS} dimensions"
                    )
            except (KeyError, TypeError, ValueError) as error:
                self.send_error(400, str(error))
                return

            # Both arrays have already been limited to 20 dimensions. A dot
            # product gives one similarity score for every indexed document.
            scores = document_vectors @ query_vector[:NUM_DIMENSIONS]
            ranked_indexes = np.argsort(-scores, kind="stable")

            output = StringIO()
            writer = csv.writer(output, lineterminator="\n")
            for rank, document_index in enumerate(ranked_indexes, start=1):
                writer.writerow([rank, query_id, doc_ids[document_index]])

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
    args = parser.parse_args(argv)

    doc_ids, document_vectors = load_index(args.index)
    handler = make_query_handler(doc_ids, document_vectors)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)

    try:
        # The benchmark waits for this exact line before sending queries.
        print("READY", flush=True)
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
