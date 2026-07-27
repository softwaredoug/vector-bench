import h5py
import numpy as np

from exps.binaryquant import BinaryQuantVectorIndex


def test_binary_quantization_centers_each_coordinate(tmp_path):
    index_path = tmp_path / "embeddings.h5"
    vectors = np.array([[10, 10], [11, 9]], dtype=np.float64)
    with h5py.File(index_path, "w") as index_file:
        index_file.create_dataset("doc_ids", data=[b"doc-a", b"doc-b"])
        index_file.create_dataset("vectors", data=vectors)

    with h5py.File(index_path, "r") as index_file:
        index = BinaryQuantVectorIndex.index(
            index_file["doc_ids"], index_file["vectors"], dimensions=2
        )

    assert np.array_equal(
        index.means,
        np.array([10.5, 9.5]),
    )
    assert np.array_equal(
        index.packed_index,
        np.stack([np.packbits([0, 1]), np.packbits([1, 0])]),
    )
