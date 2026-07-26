"""A deliberately simple brute-force vector search application performing a random rotation (turboquant)."""


from dataclasses import dataclass
import sys

import h5py
import numpy as np
from tqdm import tqdm

from .serve import serve


# The benchmark provides full-size embeddings, but this demo intentionally
# uses only a small prefix so the search implementation stays straightforward.
DEFAULT_DIMENSIONS = 60

# Number of samples to PCA
NUM_SAMPLES = 100_000


def random_rotation(dims: int) -> np.ndarray:
    """Perform Principal Component Analysis to reduce the dimensions of the vector."""
    # Generate a random orthogonal matrix using QR decomposition
    random_matrix = np.random.randn(dims, dims)
    q, _ = np.linalg.qr(random_matrix)
    return q


@dataclass
class TurboQuantIndex:
    """The in-memory document IDs and vectors used by the search server."""

    doc_ids: list[str]
    doc_vectors: np.ndarray
    rotation: np.ndarray
    dimensions: int

    @staticmethod
    def index(
        doc_ids: h5py.Dataset,
        vectors: h5py.Dataset,
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> "TurboQuantIndex":
        """Build an index from original document vectors."""
        rows, orig_dims = vectors.shape
        dimensions = orig_dims

        if orig_dims < dimensions:
            raise ValueError(f"vectors must contain at least {dimensions} dimensions")

        rotation = random_rotation(orig_dims)

        rot_index = np.empty((rows, dimensions), dtype=np.float64)
        index_doc_ids = []

        for doc_id, vector in tqdm(
            zip(doc_ids, vectors), file=sys.stdout, total=rows, desc="Indexing", unit="doc"
        ):
            transformed = vector @ rotation
            index_doc_ids.append(
                doc_id.decode() if isinstance(doc_id, bytes) else str(doc_id)
            )
            rot_index[len(index_doc_ids) - 1] = transformed

        return TurboQuantIndex(
            index_doc_ids,
            doc_vectors=rot_index,
            rotation=rotation,
            dimensions=dimensions
        )

    def query(self, query_vector: np.ndarray, top_k: int | None = None):
        """Return ranked document IDs and scores for one query vector."""
        if len(query_vector) < self.dimensions:
            raise ValueError(
                f"Query vector must contain at least {self.dimensions} dimensions"
            )

        transformed = query_vector @ self.rotation

        scores = self.doc_vectors @ transformed
        ranked_indexes = np.argsort(-scores, kind="stable")
        if top_k is not None:
            ranked_indexes = ranked_indexes[:top_k]
        return [
            (
                rank,
                self.doc_ids[int(document_index)],
                float(scores[int(document_index)]),
            )
            for rank, document_index in enumerate(ranked_indexes, start=1)
        ]


def main(argv=None) -> None:
    """Run the naive index with the shared standalone server."""
    serve(TurboQuantIndex, argv)


if __name__ == "__main__":
    main()
