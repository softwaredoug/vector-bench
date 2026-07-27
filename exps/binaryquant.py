"""A deliberately simple brute-force vector search application performing binary quant (just signs)."""

from dataclasses import dataclass
import sys

import h5py
import numpy as np
from tqdm import tqdm

from .serve import MAX_TOP_K, serve


NUM_SAMPLES = 10000


@dataclass
class BinaryQuantVectorIndex:
    """The in-memory document IDs and vectors used by the search server."""

    doc_ids: list[str]
    packed_index: np.ndarray
    means: np.ndarray
    dimensions: int

    @staticmethod
    def index(
        doc_ids: h5py.Dataset,
        vectors: h5py.Dataset,
        dimensions: int = 0,
    ) -> "BinaryQuantVectorIndex":
        """Build an index from original document vectors."""
        rows, orig_dims = vectors.shape

        if orig_dims < dimensions:
            raise ValueError(f"vectors must contain at least {dimensions} dimensions")

        means = np.asarray(vectors).mean(axis=0)
        all_packed = []
        index_doc_ids = []

        for doc_id, vector in tqdm(
            zip(doc_ids, vectors), file=sys.stdout, total=rows, desc="Indexing", unit="doc"
        ):
            transformed = vector - means >= 0
            transformed = transformed.astype(np.uint8)
            packed = np.packbits(transformed)
            index_doc_ids.append(
                doc_id.decode() if isinstance(doc_id, bytes) else str(doc_id)
            )
            all_packed.append(packed)
        all_packed = np.stack(all_packed, axis=0)

        return BinaryQuantVectorIndex(
            index_doc_ids,
            packed_index=all_packed,
            means=means,
            dimensions=dimensions
        )

    def query(self, query_vector: np.ndarray, top_k: int | None = MAX_TOP_K):
        """Return ranked document IDs and scores for one query vector."""
        transformed = query_vector[: len(self.means)] - self.means >= 0
        packed = np.packbits(transformed.astype(np.uint8))
        # XOR + hamming
        xord = np.bitwise_xor(self.packed_index, packed)
        scores = np.bitwise_count(xord).sum(axis=1)  # count the number of differing bits

        # transformed = np.clip(np.rint(transformed), -127, 127)  # this loses information, perhaps needlessly at query time
        # print(f"Query transformation took {perf_counter() - start:.6f} seconds")
        # print(f"Query scoring took {perf_counter() - start:.6f} seconds")
        ranked_indexes = np.argsort(scores, kind="stable")
        # print(f"Query ranking took {perf_counter() - start:.6f} seconds")
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
    serve(BinaryQuantVectorIndex, argv)


if __name__ == "__main__":
    main()
