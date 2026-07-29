from unittest.mock import patch

import h5py
import numpy as np

from exps.turboquant import TurboQuantIndex


def test_turboquant_centers_before_rotation_and_binary_quantization(tmp_path):
    index_path = tmp_path / "embeddings.h5"
    vectors = np.array([[10, 10], [11, 9]], dtype=np.float32)
    with h5py.File(index_path, "w") as index_file:
        index_file.create_dataset("doc_ids", data=[b"doc-a", b"doc-b"])
        index_file.create_dataset("vectors", data=vectors)

    with h5py.File(index_path, "r") as index_file:
        rotation = np.array([[0, 1], [1, 0]], dtype=np.float32)
        projections = np.eye(2)
        with (
            patch("exps.turboquant.random_rotation", return_value=rotation),
            patch(
                "exps.turboquant.random_projection_matrix",
                return_value=projections,
            ),
        ):
            index = TurboQuantIndex.index(index_file["doc_ids"], index_file["vectors"])

    assert np.array_equal(index.means, np.array([9.5, 10.5]))
    assert np.array_equal(
        index.packed_index,
        np.stack([np.packbits([1, 0, 1, 0]), np.packbits([0, 1, 0, 1])]),
    )
