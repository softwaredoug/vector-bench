"""A deliberately simple brute-force vector search application performing a random rotation (turboquant)."""


from dataclasses import dataclass
import argparse
import sys
from collections.abc import Sequence
from typing import ClassVar

import h5py
import numpy as np
from tqdm import tqdm

from .isotropy import buffered_limits, coordinate_variances, graph_coords, graph_eigen
from .serve import MAX_TOP_K, serve


# The benchmark provides full-size embeddings, but this demo intentionally
# uses only a small prefix so the search implementation stays straightforward.
DEFAULT_DIMENSIONS = 60

# Number of vectors sampled for isotropy graphs
NUM_SAMPLES = 10_000


def random_rotation(dims: int) -> np.ndarray:
    """Perform Principal Component Analysis to reduce the dimensions of the vector."""
    # Generate a random orthogonal matrix using QR decomposition
    random_matrix = np.random.randn(dims, dims).astype(np.float32)
    q, _ = np.linalg.qr(random_matrix)
    return q


def random_projection_matrix(dims: int) -> np.ndarray:
    """Generate one random projection vector for each output dimension."""
    return np.random.randn(dims, dims).astype(np.float32)


class ProductQuantization:

    codebooks: list[np.ndarray]


def print_min_max(vectors):
    orig_dims = vectors.shape[1]
    print("dim, min, max, perc_gte_0, perc_gte_mean")
    gte0s = []
    gtemeans = []
    for dim in range(0, orig_dims, 10):
        num_gte_0 = np.sum(vectors[:, dim] > 0)
        num_lt_0 = np.sum(vectors[:, dim] < 0)
        perc_gte_0 = num_gte_0 / (num_gte_0 + num_lt_0)
        mean = np.mean(vectors[:, dim])
        num_gte_mean = np.sum(vectors[:, dim] > mean)
        num_lt_mean = np.sum(vectors[:, dim] < mean)
        perc_gte_mean = num_gte_mean / (num_gte_mean + num_lt_mean)
        gte0s.append(perc_gte_0)
        gtemeans.append(perc_gte_mean)
        print(dim, vectors[:, dim].min(), vectors[:, dim].max(), perc_gte_0, perc_gte_mean)
    mean_gte0 = np.mean(gte0s)
    mean_gtemean = np.mean(gtemeans)
    var_gte0 = np.var(gte0s)
    var_gtemean = np.var(gtemeans)
    print("mean_gte0", mean_gte0, "mean_gtemean", mean_gtemean)
    print("var_gte0", var_gte0, "var_gtemean", var_gtemean)


@dataclass
class TurboQuantIndex:
    """The in-memory document IDs and vectors used by the search server."""

    doc_ids: list[str]
    packed_index: np.ndarray
    means: np.ndarray
    rotation: np.ndarray
    projections: np.ndarray
    dimensions: int
    graph_isotropy: ClassVar[bool] = False

    @staticmethod
    def index(
        doc_ids: h5py.Dataset,
        vectors: h5py.Dataset,
    ) -> "TurboQuantIndex":
        """Build an index from original document vectors."""
        rows, orig_dims = vectors.shape
        dimensions = orig_dims

        if orig_dims < dimensions:
            raise ValueError(f"vectors must contain at least {dimensions} dimensions")

        rotation = random_rotation(orig_dims)
        projections = random_projection_matrix(orig_dims)

        if TurboQuantIndex.graph_isotropy:
            sample = vectors[: min(NUM_SAMPLES, rows)]
            print("orig")
            print_min_max(sample)
            rotated_sample = sample @ rotation
            print("rot")
            print_min_max(rotated_sample)
            coordinate_values = np.concatenate(
                [coordinate_variances(sample), coordinate_variances(rotated_sample)]
            )
            coordinate_limits = buffered_limits(coordinate_values)
            graph_coords(
                sample, "graph_coord_before.png", y_limits=coordinate_limits
            )
            graph_eigen(sample, "graph_eigen_before.png")
            graph_coords(
                rotated_sample, "graph_coord_after.png", y_limits=coordinate_limits
            )
            graph_eigen(rotated_sample, "graph_eigen_after.png")

        rotated_vectors = np.empty((rows, orig_dims), dtype=np.float32)
        means = np.zeros(orig_dims, dtype=np.float32)
        index_doc_ids = []

        for row, (doc_id, vector) in enumerate(tqdm(
            zip(doc_ids, vectors), file=sys.stdout, total=rows, desc="Indexing", unit="doc"
        )):
            transformed = vector @ rotation
            rotated_vectors[row] = transformed
            means += transformed
            index_doc_ids.append(
                doc_id.decode() if isinstance(doc_id, bytes) else str(doc_id)
            )

        means /= rows
        packed_index = np.empty(
            (rows, (2 * orig_dims + 7) // 8), dtype=np.uint8
        )
        for row, rotated_vector in enumerate(rotated_vectors):
            centered = rotated_vector - means
            bits = np.concatenate(
                [centered >= 0, centered @ projections >= 0]
            )
            packed_index[row] = np.packbits(bits)

        return TurboQuantIndex(
            index_doc_ids,
            packed_index=packed_index,
            means=means,
            rotation=rotation,
            projections=projections,
            dimensions=dimensions
        )

    def query(
        self, query_vector: np.ndarray, top_k: int | None = MAX_TOP_K
    ) -> list[tuple[int, str, float]]:
        """Return ranked document IDs and scores for one query vector."""
        if len(query_vector) < self.dimensions:
            raise ValueError(
                f"Query vector must contain at least {self.dimensions} dimensions"
            )

        transformed = query_vector[: len(self.means)] @ self.rotation - self.means
        packed_query = np.packbits(
            np.concatenate([transformed >= 0, transformed @ self.projections >= 0])
        )

        distances = np.bitwise_count(
            np.bitwise_xor(self.packed_index, packed_query)
        ).sum(axis=1)
        ranked_indexes = np.argsort(distances, kind="stable")
        if top_k is not None:
            ranked_indexes = ranked_indexes[:top_k]
        return [
            (
                rank,
                self.doc_ids[int(document_index)],
                float(distances[int(document_index)]),
            )
            for rank, document_index in enumerate(ranked_indexes, start=1)
        ]


def main(argv: Sequence[str] | None = None) -> None:
    """Run TurboQuant with the shared standalone server."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--graph-isotropy", action="store_true")
    args, serve_argv = parser.parse_known_args(argv)
    TurboQuantIndex.graph_isotropy = args.graph_isotropy
    serve(TurboQuantIndex, serve_argv)


if __name__ == "__main__":
    main()
