"""Tools for graphing the covariance eigenvalues of vector data."""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def coordinate_variances(vectors: np.ndarray) -> np.ndarray:
    """Return the variance of each vector coordinate."""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    covariance = np.cov(centered, rowvar=False)
    return np.diag(covariance)


def buffered_limits(values: np.ndarray) -> tuple[float, float]:
    """Return min/max values expanded by a small plotting buffer."""
    minimum = float(values.min())
    maximum = float(values.max())
    buffer = max((maximum - minimum) * 0.05, abs(maximum) * 0.05, 1e-12)
    return minimum - buffer, maximum + buffer


def graph_eigen(vectors: np.ndarray, output_path: str | Path) -> None:
    """Graph all covariance eigenvalues for a set of vectors."""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]

    figure, axis = plt.subplots()
    axis.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues)
    axis.set_xlabel("Eigenvalue rank")
    axis.set_ylabel("Eigenvalue")
    axis.set_title("Vector covariance eigenvalues")
    for index in range(24, len(eigenvalues), 25):
        axis.annotate(
            f"{eigenvalues[index]:.3g}",
            (index + 1, eigenvalues[index]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )
    figure.savefig(output_path)
    plt.close(figure)


def graph_coords(
    vectors: np.ndarray,
    output_path: str | Path,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Graph the covariance variance for each coordinate."""
    variances = coordinate_variances(vectors)
    minimum_index = int(np.argmin(variances))
    maximum_index = int(np.argmax(variances))
    if y_limits is None:
        y_limits = buffered_limits(variances)

    figure, axis = plt.subplots()
    axis.plot(np.arange(1, len(variances) + 1), variances)
    axis.set_ylim(*y_limits)
    axis.set_xlabel("Coordinate")
    axis.set_ylabel("Variance")
    axis.set_title("Coordinate variances")
    axis.annotate(
        f"min: {variances[minimum_index]:.3g}",
        (minimum_index + 1, variances[minimum_index]),
        textcoords="offset points",
        xytext=(0, 6),
        ha="center",
    )
    axis.annotate(
        f"max: {variances[maximum_index]:.3g}",
        (maximum_index + 1, variances[maximum_index]),
        textcoords="offset points",
        xytext=(0, 6),
        ha="center",
    )
    figure.savefig(output_path)
    plt.close(figure)
