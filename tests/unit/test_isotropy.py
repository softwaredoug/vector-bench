from unittest.mock import Mock, patch

import numpy as np

from exps.isotropy import graph_coords, graph_eigen


def test_graph_annotates_every_25th_eigenvalue(tmp_path):
    axis = Mock()
    figure = Mock()
    vectors = np.eye(50, dtype=np.float32)
    with patch("exps.isotropy.plt.subplots", return_value=(figure, axis)):
        graph_eigen(vectors, tmp_path / "graph.png")

    annotations = axis.annotate.call_args_list
    eigenvalues = np.linalg.eigvalsh(np.cov(vectors, rowvar=False))[::-1]
    assert len(annotations) == 2
    assert annotations[0].args[0] == f"{eigenvalues[24]:.3g}"
    assert annotations[0].args[1][0] == 25
    assert np.isclose(annotations[0].args[1][1], eigenvalues[24])
    assert annotations[1].args[1][0] == 50


def test_graph_coords_labels_min_max_and_uses_y_limits(tmp_path):
    axis = Mock()
    figure = Mock()
    vectors = np.array([[0, 0], [1, 2], [2, 4]], dtype=np.float32)
    with patch("exps.isotropy.plt.subplots", return_value=(figure, axis)):
        graph_coords(vectors, tmp_path / "graph.png", y_limits=(1.0, 5.0))

    variances = np.diag(np.cov(vectors, rowvar=False))
    axis.set_ylim.assert_called_once_with(1.0, 5.0)
    assert [call.args[0] for call in axis.annotate.call_args_list] == [
        f"min: {variances.min():.3g}",
        f"max: {variances.max():.3g}",
    ]
