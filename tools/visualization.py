"""Matplotlib visualization tools for DeEntropy V1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.schemas import EmbeddingResult


def plot_embedding(
    embedding_result: EmbeddingResult,
    output_dir: str | Path,
    labels: Any = None,
    title: str | None = None,
) -> Path:
    """Save a 2D scatterplot for an embedding and return the image path."""
    if not embedding_result.success:
        raise ValueError("Cannot plot an unsuccessful embedding.")

    embedding = _as_embedding_array(embedding_result.embedding)
    if embedding.shape[1] < 2:
        raise ValueError("Embedding scatterplot requires at least two components.")

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / f"{embedding_result.method}_embedding.png"

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    x_values = embedding[:, 0]
    y_values = embedding[:, 1]

    if labels is None:
        ax.scatter(x_values, y_values, s=28, alpha=0.82, edgecolors="none")
    else:
        label_values = _as_label_array(labels, embedding.shape[0])
        unique_labels = pd.Series(label_values).astype("category")
        codes = unique_labels.cat.codes.to_numpy()
        scatter = ax.scatter(
            x_values,
            y_values,
            c=codes,
            s=28,
            alpha=0.86,
            cmap="tab10",
            edgecolors="none",
        )
        handles, _ = scatter.legend_elements()
        categories = [str(item) for item in unique_labels.cat.categories.tolist()]
        ax.legend(
            handles,
            categories,
            title="Label",
            loc="best",
            fontsize="small",
            title_fontsize="small",
        )

    ax.set_title(title or f"{embedding_result.method} embedding")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linewidth=0.4, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_pca_explained_variance(
    embedding_result: EmbeddingResult,
    output_dir: str | Path,
) -> Path:
    """Save a PCA explained-variance plot from embedding metadata."""
    if embedding_result.method != "pca":
        raise ValueError("PCA explained-variance plots require method='pca'.")
    ratios = embedding_result.metadata.get("explained_variance_ratio")
    if not ratios:
        raise ValueError("PCA result does not contain explained_variance_ratio metadata.")

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / "pca_variance.png"

    ratio_array = np.asarray(ratios, dtype=float)
    components = np.arange(1, len(ratio_array) + 1)
    cumulative = np.cumsum(ratio_array)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.bar(components, ratio_array, alpha=0.78, label="Individual")
    ax.plot(components, cumulative, marker="o", linewidth=1.8, label="Cumulative")
    ax.set_title("PCA explained variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_xticks(components)
    ax.set_ylim(0, min(1.05, max(1.0, float(cumulative[-1]) + 0.05)))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_embedding_plots(
    embedding_results: list[EmbeddingResult],
    output_dir: str | Path,
    labels: Any = None,
) -> list[Path]:
    """Generate standard plots for successful embeddings."""
    paths: list[Path] = []
    for result in embedding_results:
        if not result.success:
            continue
        embedding = _as_embedding_array(result.embedding)
        if embedding.shape[1] >= 2:
            paths.append(plot_embedding(result, output_dir, labels=labels))
        if result.method == "pca" and result.metadata.get("explained_variance_ratio"):
            paths.append(plot_pca_explained_variance(result, output_dir))
    return paths


def _as_embedding_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Embedding must be a two-dimensional numeric matrix.")
    if not np.isfinite(array).all():
        raise ValueError("Embedding contains NaN or infinite values.")
    return array


def _as_label_array(labels: Any, expected_rows: int) -> np.ndarray:
    if isinstance(labels, pd.Series):
        values = labels.to_numpy()
    else:
        values = np.asarray(labels)
    if values.shape[0] != expected_rows:
        raise ValueError("Labels must have the same number of rows as the embedding.")
    return values
