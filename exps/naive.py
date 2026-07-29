"""A deliberately simple brute-force vector search application."""

from dataclasses import dataclass
import sys

import h5py
import numpy as np
from tqdm import tqdm

from .serve import MAX_TOP_K, serve


@dataclass
class VectorIndex:
    """The in-memory document IDs and vectors used by the search server."""

    doc_ids: list[str]
    doc_vectors: np.ndarray

    @staticmethod
    def index(
        doc_ids: h5py.Dataset,
        vectors: h5py.Dataset,
    ) -> "VectorIndex":
        """Build an index from original document vectors."""

        rows, orig_dims = vectors.shape

        index = np.empty((rows, orig_dims), dtype=np.float32)
        index_doc_ids = []

        for doc_id, vector in tqdm(
            zip(doc_ids, vectors), file=sys.stdout, total=rows, desc="Indexing", unit="doc"
        ):
            index_doc_ids.append(
                doc_id.decode() if isinstance(doc_id, bytes) else str(doc_id)
            )
            index[len(index_doc_ids) - 1] = vector.astype(np.float32)

        return VectorIndex(
            index_doc_ids,
            doc_vectors=index,
        )

    def query(self, query_vector: np.ndarray, top_k: int | None = MAX_TOP_K):
        """Return ranked document IDs and scores for one query vector."""

        scores = self.doc_vectors @ query_vector.astype(np.float32, copy=False)
        if top_k is not None and 0 < top_k < len(scores):
            ranked_indexes = np.argpartition(scores, -top_k)[-top_k:]
            ranked_indexes = ranked_indexes[
                np.argsort(-scores[ranked_indexes], kind="stable")
            ]
        else:
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
    serve(VectorIndex, argv)


if __name__ == "__main__":
    main()
