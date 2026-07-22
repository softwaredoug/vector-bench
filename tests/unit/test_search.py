from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from vector_bench.search import search


@patch("vector_bench.search._post_query", return_value=["doc-a", "doc-b"])
def test_search_prints_query_and_average_metrics(mock_post_query):
    output = StringIO()

    search(
        SimpleNamespace(port=1234),
        ["q1"],
        np.array([[1, 2]], dtype=np.float32),
        {"q1": ["doc-a", "doc-c"]},
        top_k=2,
        output=output,
    )

    rows = output.getvalue().splitlines()
    assert rows[0].split(",")[0] == "q1"
    assert float(rows[0].split(",")[2]) == 0.5
    assert rows[1].startswith(",")
    assert float(rows[1].split(",")[2]) == 0.5
    mock_post_query.assert_called_once()
    call_args = mock_post_query.call_args.args
    assert call_args[:2] == (1234, "q1")
    np.testing.assert_array_equal(call_args[2], np.array([1, 2]))


def test_search_rejects_mismatched_query_ids_and_vectors():
    output = StringIO()

    with pytest.raises(ValueError, match="same length"):
        search(
            SimpleNamespace(port=1234),
            ["q1"],
            np.array([[1], [2]], dtype=np.float32),
            {},
            top_k=1,
            output=output,
        )
