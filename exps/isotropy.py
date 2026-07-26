"""Tools for graphing the covariance eigenvalues of vector data."""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


def graph_coords(vectors: np.ndarray, output_path: str | Path) -> None:
    """Graph the covariance variance for each coordinate."""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    covariance = np.cov(centered, rowvar=False)
    variances = np.diag(covariance)

    figure, axis = plt.subplots()
    axis.plot(np.arange(1, len(variances) + 1), variances)
    axis.set_xlabel("Coordinate")
    axis.set_ylabel("Variance")
    axis.set_title("Coordinate variances")
    figure.savefig(output_path)
    plt.close(figure)
