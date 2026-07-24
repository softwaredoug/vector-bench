"""A deliberately simple brute-force vector search application.

This application is separate from the benchmark and communicates with it only
through the HDF5 index format and HTTP protocol documented in the project
README. It is also intended to be readable as a student starter application.
"""

import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
import sys
from collections.abc import Generator
from urllib.parse import parse_qs

import h5py
import numpy as np
from tqdm import tqdm


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


# The benchmark provides full-size embeddings, but this demo intentionally
# uses only a small prefix so the search implementation stays straightforward.
DEFAULT_DIMENSIONS = 60


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
            raise ValueError(f"vectors must contain at least {dimensions} dimensions")

        index = np.empty((rows, dimensions), dtype=np.float64)
        index_doc_ids = []

        for doc_id, vector in tqdm(
            zip(doc_ids, vectors), file=sys.stdout, total=rows, desc="Indexing", unit="doc"
        ):
            index_doc_ids.append(
                doc_id.decode() if isinstance(doc_id, bytes) else str(doc_id)
            )
            index[len(index_doc_ids) - 1] = vector[:dimensions]

        return VectorIndex(
            index_doc_ids,
            doc_vectors=index,
            dimensions=dimensions,
        )

    def query(self, query_vector: np.ndarray, top_k: int | None = None):
        """Return ranked document IDs for one query vector."""
        if len(query_vector) < self.dimensions:
            raise ValueError(
                f"Query vector must contain at least {self.dimensions} dimensions"
            )

        scores = self.doc_vectors @ query_vector[: self.dimensions]
        ranked_indexes = np.argsort(-scores, kind="stable")
        if top_k is not None:
            ranked_indexes = ranked_indexes[:top_k]
        return [
            (rank, self.doc_ids[int(document_index)])
            for rank, document_index in enumerate(ranked_indexes, start=1)
        ]


def load_index(index_path: Path, dimensions: int = DEFAULT_DIMENSIONS) -> VectorIndex:
    """Read the HDF5 embeddings and build the search index."""
    with datasets(index_path) as (doc_ids, vectors):
        return VectorIndex.index(doc_ids, vectors, dimensions=dimensions)


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
    print("Inedx loaded")
    handler = make_query_handler(vector_index)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)

    try:
        print("READY", flush=True)
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
