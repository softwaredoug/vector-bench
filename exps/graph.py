import numpy as np
from tqdm import tqdm
import h5py

from dataclasses import dataclass
from .serve import MAX_TOP_K, serve
from .utils.heap import MaxHeap
import sys


DEFAULT_M = 16   # Max connections
DEFAULT_EF_CONSTRUCTION = 4  # Size of the dynamic list for the nearest neighbors during construction
DEFAULT_EF_SEARCH = 64  # Size of the dynamic list for the nearest neighbors during searching


def beam_search(query: np.ndarray,
                vectors: np.ndarray,
                adjacencies: list[list[int]],
                ef: int = DEFAULT_EF_CONSTRUCTION,
                num_brute_force: int = 100) -> list[tuple[np.float32, int]]:

    # Seed with num_brute_force
    sims = vectors[:num_brute_force] @ query
    seed_idxs = np.argsort(sims)[-ef:]  # Get the indices of the top ef similarities

    exploration_frontier: MaxHeap = MaxHeap(heap=[(np.float32(sims[idx]), idx) for idx in seed_idxs],
                                            max_size=ef)
    visited = set(seed_idxs)

    adding = True

    while adding:
        adding = False
        for _, idx in exploration_frontier.items():
            adjacent_to_idx = adjacencies[idx]
            adjacent_vects = vectors[adjacent_to_idx]
            sims = adjacent_vects @ query

            for idx, sim in zip(adjacent_to_idx, sims):
                if idx in visited:
                    continue

                exploration_frontier.pushpop((sim, idx))
                print(f"Frontier now len: {len(exploration_frontier)}")
                visited.add(idx)
                adding = True

    return exploration_frontier.sorted


@dataclass
class GraphIndex:

    doc_ids: list[str]
    vectors: np.ndarray
    adjacencies: list[list[int]]

    @staticmethod
    def index(
        doc_ids: h5py.Dataset,
        vectors: h5py.Dataset,
    ) -> "GraphIndex":
        rows, _ = vectors.shape

        all_doc_ids = []
        all_vectors = np.empty((rows, vectors.shape[1]), dtype=np.float32)
        adjacencies = []

        for doc_id, vector in tqdm(
            zip(doc_ids, vectors), file=sys.stdout, total=rows, desc="Indexing", unit="doc"
        ):
            doc_id = doc_id.decode() if isinstance(doc_id, bytes) else str(doc_id)
            all_doc_ids.append(doc_id)
            all_vectors[len(all_doc_ids) - 1] = vector.astype(np.float32)
            curr_idx = len(all_doc_ids) - 1
            adjacencies.append([])

            curr_adjacents = beam_search(vector, all_vectors[:len(all_doc_ids)],
                                         adjacencies,
                                         ef=DEFAULT_EF_CONSTRUCTION + 1)
            # Connect all adjacent nodes to the new node and vice versa
            for _, idx in curr_adjacents:
                if idx == curr_idx:
                    continue
                if idx < len(adjacencies):
                    adjacencies[idx].append(curr_idx)
                    adjacencies[curr_idx].append(idx)

        return GraphIndex(
            doc_ids=all_doc_ids,
            vectors=all_vectors,
            adjacencies=adjacencies
        )

    def query(
        self, query_vector: np.ndarray, top_k: int | None = MAX_TOP_K
    ) -> list[tuple[int, str, float]]:
        """Beam search from root to get nearest neighbors."""
        if top_k is None or top_k <= 0:
            top_k = MAX_TOP_K

        top_nodes = beam_search(query_vector,
                                vectors=self.vectors,
                                adjacencies=self.adjacencies, ef=DEFAULT_EF_SEARCH)
        results = []
        for rank, (score, idx) in enumerate(top_nodes):
            doc_id = self.doc_ids[idx]
            results.append((rank, doc_id, score))
        return results


def main(argv=None) -> None:
    """Run the naive index with the shared standalone server."""
    serve(GraphIndex, argv)


if __name__ == "__main__":
    main()
