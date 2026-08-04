import heapq
import numpy as np
from tqdm import tqdm
import h5py

from dataclasses import dataclass, field
from .serve import MAX_TOP_K, serve
import sys


DEFAULT_M = 16   # Max connections
DEFAULT_EF_CONSTRUCTION = 4  # Size of the dynamic list for the nearest neighbors during construction
DEFAULT_EF_SEARCH = 64  # Size of the dynamic list for the nearest neighbors during searching


def cos_dist(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Get a distance between two vectors based on cosine similarity, assuming they're normalized."""
    dotted = np.dot(vector1, vector2)
    assert -1.0 <= dotted <= 1.0, f"Dot product {dotted} is out of range [-1, 1]"
    return float(1.0 - np.dot(vector1, vector2))


@dataclass
class Node:
    vector: np.ndarray
    doc_id: str
    max_neighbors: int
    neighbors: list[tuple[float, "Node"]] = field(default_factory=list)

    def add_neighbor(self, node):
        dist = cos_dist(self.vector, node.vector)
        heapq.heappush(self.neighbors, (dist, node))
        if len(self.neighbors) > self.max_neighbors:
            heapq.heappop(self.neighbors)

    @property
    def sims(self):
        return [np.dot(self.vector, n[1].vector) for n in self.neighbors]

    @property
    def nodes(self):
        return [n[1] for n in self.neighbors]

    def __repr__(self):
        neighbor_dists = [float(n[0]) for n in self.neighbors]
        neighbor_sims = [float(np.dot(self.vector, n[1].vector)) for n in self.neighbors]
        neighbor_docids = [n[1].doc_id for n in self.neighbors]
        return f"Node({self.doc_id}, dists={neighbor_dists}, sims={neighbor_sims}, doc_ids={neighbor_docids})"

    def __eq__(self, other):
        return self.doc_id == other.doc_id

    def __lt__(self, other):
        return hash(self.doc_id) < hash(other.doc_id)


def beam_search(query: np.ndarray, root: Node,
                ef: int = DEFAULT_EF_CONSTRUCTION) -> list[tuple[float, Node]]:

    exploration_frontier = [(cos_dist(query, root.vector), root)]
    visited = set()

    adding = True

    while adding:
        adding = False
        for _, node in exploration_frontier:
            for neighbor in node.nodes:
                if neighbor.doc_id in visited:
                    continue

                dist_to_query = cos_dist(query, neighbor.vector)
                heapq.heappush(exploration_frontier, (dist_to_query, neighbor))
                print(f"Tracking {len(exploration_frontier)} neighbors for node {neighbor.doc_id}")
                visited.add(neighbor.doc_id)
                adding = True
        # Truncate to ef closest
        print("Truncating...")
        exploration_frontier = heapq.nsmallest(ef, exploration_frontier)

    return exploration_frontier


def select_best_connections(query: np.ndarray, ef: list[tuple[float, Node]], m: int):
    """Most useful m connections out of ef candidates based on distance to query."""
    best_neighbors = []
    for dist, candidate in ef:
        if len(best_neighbors) >= m:
            break
        # Check if candidate is closer than any of the best neighbors
        if all(cos_dist(candidate.vector, neighbor.vector) > dist for _, neighbor in best_neighbors):
            best_neighbors.append((dist, candidate))
    return best_neighbors


@dataclass
class GraphIndex:

    doc_ids: list[str]
    root: Node | None = None

    @staticmethod
    def index(
        doc_ids: h5py.Dataset,
        vectors: h5py.Dataset,
    ) -> "GraphIndex":
        rows, _ = vectors.shape

        root = None
        all_doc_ids = []
        all_vectors = np.empty((rows, vectors.shape[1]), dtype=np.float32)

        for doc_id, vector in tqdm(
            zip(doc_ids, vectors), file=sys.stdout, total=rows, desc="Indexing", unit="doc"
        ):
            doc_id = doc_id.decode() if isinstance(doc_id, bytes) else str(doc_id)
            all_doc_ids.append(doc_id)
            all_vectors[len(all_doc_ids) - 1] = vector.astype(np.float32)
            if root is None:
                root = Node(vector.astype(np.float32), doc_id, DEFAULT_M)
            else:
                new_node = Node(vector.astype(np.float32), doc_id, DEFAULT_M)
                # Add new node to the graph
                ef_frontier = beam_search(new_node.vector, root, ef=DEFAULT_EF_CONSTRUCTION)
                print(f"Found {len(ef_frontier)} neighbors for node {doc_id}")
                best = select_best_connections(new_node.vector, ef_frontier, DEFAULT_M)
                for _, candidate in best:
                    new_node.add_neighbor(candidate)
                    candidate.add_neighbor(new_node)
        return GraphIndex(
            doc_ids=all_doc_ids,
            root=root
        )

    def query(self, query_vector: np.ndarray, top_k: int | None = MAX_TOP_K):
        """Beam search from root to get nearest neighbors."""
        if self.root is None:
            raise ValueError("Graph index is empty. Please index the data first.")

        if top_k is None or top_k <= 0:
            top_k = MAX_TOP_K

        return beam_search(query_vector, self.root, ef=DEFAULT_EF_SEARCH)


def main(argv=None) -> None:
    """Run the naive index with the shared standalone server."""
    serve(GraphIndex, argv)


if __name__ == "__main__":
    main()
