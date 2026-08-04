from exps.graph import beam_search
from exps.utils.sim import norm

import math

import numpy as np


def assert_sim_order(query_vector, results, vectors):
    """Confirm sort order from most to least similar."""
    for earlier_result, next_result in zip(results, results[1:]):
        earlier_sim = earlier_result[0]
        next_sim = next_result[0]
        assert earlier_result[0] >= next_result[0], f"Results are not sorted by similarity: {earlier_sim} > {next_sim}"
        # Now confirm the sim
        earlier_vector = vectors[earlier_result[1]]
        next_vector = vectors[next_result[1]]
        actual_earlier_sim = earlier_vector @ query_vector
        actual_next_sim = next_vector @ query_vector
        assert math.isclose(earlier_sim, actual_earlier_sim, rel_tol=1e-5), f"Earlier sim {earlier_sim} does not match actual {actual_earlier_sim}"
        assert math.isclose(next_sim, actual_next_sim, rel_tol=1e-5), f"Next sim {next_sim} does not match actual {actual_next_sim}"


def test_beam_search_base():
    # Create a simple graph with 3 nodes

    vectors = norm([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
    adjacencies = [[1], [0, 2], [1]]

    # Perform beam search with a query vector
    query_vector = np.array([0.5, 0.5])
    ef = 2
    results = beam_search(query_vector, vectors, adjacencies, ef)

    assert_sim_order(query_vector, results, vectors)

    # Check that the results are sorted by distance and contain the expected nodes
    assert len(results) == 2


def test_beam_search_long_chain():
    adjacencies = [[] for _ in range(255)]
    vectors = np.empty((255, 2), dtype=np.float32)
    for idx in range(255):
        x = math.cos(2 * math.pi * idx / 255)
        y = math.sin(2 * math.pi * idx / 255)
        vector = norm(np.array([x, y]))
        adjacencies[idx - 1] = [idx]
        vectors[idx] = vector

    # Perform beam search with a query vector
    query_vector = np.array([0.5, 0.5])
    ef = 5
    results = beam_search(query_vector, vectors, adjacencies, ef)
    assert len(results) == 5
    assert_sim_order(query_vector, results, vectors)
