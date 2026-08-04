from exps.graph import beam_search, Node

import numpy as np
import math


def normed(vect: np.ndarray) -> np.ndarray:
    """Return a normalized vector."""
    norm = np.linalg.norm(vect)
    if norm == 0:
        return vect
    return vect / norm


def assert_sim_order(query, results):
    """Confirm sort order from most to least similar."""
    earlier_sim = 1.1
    for result in results:
        curr_sim = np.dot(result[1].vector, query)
        assert earlier_sim >= curr_sim, f"Results are not sorted by similarity: {earlier_sim} > {curr_sim}"
        earlier_sim = curr_sim


def test_beam_search_base():
    # Create a simple graph with 3 nodes
    node_a = Node(normed(np.array([1.0, 0.0])), "doc-a", 2)
    node_b = Node(normed(np.array([1.0, 1.0])), "doc-b", 2)
    node_c = Node(normed(np.array([-1.0, 0.0])), "doc-c", 2)

    # Connect the nodes
    node_a.add_neighbor(node_b)
    node_b.add_neighbor(node_a)
    node_b.add_neighbor(node_c)
    node_c.add_neighbor(node_b)

    # Perform beam search with a query vector
    query_vector = np.array([0.5, 0.5])
    ef = 2
    results = beam_search(query_vector, node_a, ef)

    assert_sim_order(query_vector, results)

    # Check that the results are sorted by distance and contain the expected nodes
    assert len(results) == 2
    assert results[0][1] == node_b  # Closest neighbor
    assert results[1][1] == node_a or results[1][1] == node_c  # Second closest neighbor


def test_beam_search_bad_graph():
    # Create a simple graph with 3 nodes
    node_a = Node(normed(np.array([1.0, 0.0])), "doc-a", 2)
    node_b = Node(normed(np.array([1.0, 1.0])), "doc-b", 2)
    node_c = Node(normed(np.array([-1.0, 0.0])), "doc-c", 2)

    # Connect the nodes
    node_a.add_neighbor(node_b)
    node_b.add_neighbor(node_a)

    # Perform beam search with a query vector
    query_vector = np.array([0.5, 0.5])
    ef = 2
    results = beam_search(query_vector, node_a, ef)

    assert_sim_order(query_vector, results)

    # Check that the results are sorted by distance and contain the expected nodes
    assert len(results) == 2
    assert results[0][1] == node_b  # Closest neighbor
    assert results[1][1] == node_a or results[1][1] == node_c  # Second closest neighbor


def test_beam_search_long_chain():
    nodes = []
    all_vectors = np.empty((255, 2), dtype=np.float32)
    for idx in range(255):
        x = math.cos(2 * math.pi * idx / 255)
        y = math.sin(2 * math.pi * idx / 255)
        nodes.append(Node(normed(np.array([x, y])), f"doc-{idx}", 2))
        nodes[idx - 1].add_neighbor(nodes[idx])  # Connect to previous node
        all_vectors[idx] = nodes[idx].vector

    # Perform beam search with a query vector
    query_vector = np.array([0.5, 0.5])
    results = beam_search(query_vector, nodes[0], ef=5)
    assert_sim_order(query_vector, results)
