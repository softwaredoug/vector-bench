"""A deliberately simple brute-force vector search application."""

from dataclasses import dataclass
import sys

import h5py
import numpy as np
from tqdm import tqdm

from .serve import serve


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
        """Build an index from original document vectors."""
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


def main(argv=None) -> None:
    """Run the naive index with the shared standalone server."""
    serve(VectorIndex, argv)


if __name__ == "__main__":
    main()
