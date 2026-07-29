"""A deliberately simple brute-force vector search application performing PCA."""

from dataclasses import dataclass
import sys
import os

import h5py
import numpy as np
from tqdm import tqdm

from .serve import MAX_TOP_K, serve


# The benchmark provides full-size embeddings, but this demo intentionally
# uses only a small prefix so the search implementation stays straightforward.
DEFAULT_DIMENSIONS = 60

# Number of samples to PCA
DEFAULT_NUM_SAMPLES = 100_000


def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform Principal Component Analysis to reduce the dimensions of the vector."""
    mean = X.mean(axis=0)
    centered = X - mean

    # How each dimension varies with every other dimension
    covariance = np.cov(centered, rowvar=False).astype(np.float32)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvectors[:, :n_components], mean


@dataclass
class PCAVectorIndex:
    """The in-memory document IDs and vectors used by the search server."""

    doc_ids: list[str]
    doc_vectors: np.ndarray
    pca_eigens: np.ndarray
    means: np.ndarray
    dimensions: int

    @staticmethod
    def index(
        doc_ids: h5py.Dataset,
        vectors: h5py.Dataset,
    ) -> "PCAVectorIndex":
        """Build an index from original document vectors."""
        dimensions = int(os.getenv("PCA_DIMENSIONS", DEFAULT_DIMENSIONS))
        num_samples = int(os.getenv("PCA_NUM_SAMPLES", DEFAULT_NUM_SAMPLES))
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        rows, orig_dims = vectors.shape

        if orig_dims < dimensions:
            raise ValueError(f"vectors must contain at least {dimensions} dimensions")

        pca_matrix = vectors[:num_samples]
        pca_eigens, means = pca(pca_matrix, dimensions)

        pca_index = np.empty((rows, dimensions), dtype=np.float32)
        index_doc_ids = []

        for doc_id, vector in tqdm(
            zip(doc_ids, vectors), file=sys.stdout, total=rows, desc="Indexing", unit="doc"
        ):
            transformed = vector @ pca_eigens
            index_doc_ids.append(
                doc_id.decode() if isinstance(doc_id, bytes) else str(doc_id)
            )
            pca_index[len(index_doc_ids) - 1] = transformed

        return PCAVectorIndex(
            index_doc_ids,
            doc_vectors=pca_index,
            dimensions=dimensions,
            pca_eigens=pca_eigens,
            means=means
        )

    def query(self, query_vector: np.ndarray, top_k: int | None = MAX_TOP_K):
        """Return ranked document IDs and scores for one query vector."""
        if len(query_vector) < self.dimensions:
            raise ValueError(
                f"Query vector must contain at least {self.dimensions} dimensions"
            )

        transformed = query_vector @ self.pca_eigens

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
    serve(PCAVectorIndex, argv)


if __name__ == "__main__":
    main()
