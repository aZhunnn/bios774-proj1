"""Embedding-quality evaluation tools for DeEntropy V1."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from agent.schemas import EmbeddingResult, EvaluationResult


def evaluate_embedding(
    X_original: Any,
    embedding_result: EmbeddingResult,
    labels: Any = None,
    config: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Evaluate local and global structure preservation for one embedding."""
    config = config or {}
    evaluation_config = config.get("evaluation", config)
    warnings: list[str] = []
    notes: list[str] = []
    intrinsic_metrics: dict[str, float | None] = {
        "trustworthiness": None,
        "knn_preservation": None,
        "distance_pearson_correlation": None,
        "distance_spearman_correlation": None,
    }
    external_validation_metrics: dict[str, float | None] = {}

    if not embedding_result.success:
        warnings.append(
            "Embedding was unsuccessful; quality metrics were not computed."
        )
        return EvaluationResult(
            embedding_id=embedding_result.embedding_id,
            intrinsic_metrics=intrinsic_metrics,
            external_validation_metrics=external_validation_metrics,
            metrics=intrinsic_metrics,
            warnings=warnings,
            notes=notes,
        )

    try:
        X = _as_numeric_array(X_original, name="X_original")
        embedding = _as_numeric_array(embedding_result.embedding, name="embedding")
        _validate_shapes(X, embedding)
    except Exception as exc:
        warnings.append(str(exc))
        return EvaluationResult(
            embedding_id=embedding_result.embedding_id,
            intrinsic_metrics=intrinsic_metrics,
            external_validation_metrics=external_validation_metrics,
            metrics=intrinsic_metrics,
            warnings=warnings,
            notes=notes,
        )

    n_neighbors = int(evaluation_config.get("trustworthiness_neighbors", 10))
    max_trustworthiness_neighbors = max(1, (X.shape[0] - 1) // 2)
    effective_neighbors = min(max(1, n_neighbors), max_trustworthiness_neighbors)

    if X.shape[0] < 3:
        warnings.append("At least three samples are required for trustworthiness and kNN metrics.")
    else:
        try:
            intrinsic_metrics["trustworthiness"] = float(
                trustworthiness(X, embedding, n_neighbors=effective_neighbors)
            )
        except Exception as exc:
            warnings.append(f"Trustworthiness could not be computed: {exc}")

        try:
            intrinsic_metrics["knn_preservation"] = float(
                knn_preservation(X, embedding, n_neighbors=effective_neighbors)
            )
        except Exception as exc:
            warnings.append(f"k-nearest-neighbor preservation could not be computed: {exc}")

    try:
        pearson, spearman = distance_correlations(
            X,
            embedding,
            sample_size=int(evaluation_config.get("distance_sample_size", 1000)),
            random_seed=int(config.get("random_seed", 42)),
        )
        intrinsic_metrics["distance_pearson_correlation"] = pearson
        intrinsic_metrics["distance_spearman_correlation"] = spearman
    except Exception as exc:
        warnings.append(f"Distance correlations could not be computed: {exc}")

    if embedding_result.method in {"tsne", "umap"}:
        notes.append(
            f"{embedding_result.method} emphasizes local neighborhoods; global distance correlations should be interpreted cautiously."
        )
    if embedding_result.method in {"pca", "kernel_pca", "sparse_pca"}:
        notes.append("PCA-family embeddings have method-specific variance or component metadata when available.")
    if embedding.shape[1] != 2:
        notes.append("Embedding is not 2D; scatterplot visualization may use only the first two components.")

    if labels is not None:
        external_validation_metrics = _external_validation(
            embedding,
            labels,
            n_neighbors=int(
                evaluation_config.get(
                    "external_validation_neighbors",
                    evaluation_config.get("trustworthiness_neighbors", 10),
                )
            ),
            warnings=warnings,
        )
        notes.append(
            "External validation uses labels only after embedding creation; labels were not used to fit or tune the embedding."
        )

    return EvaluationResult(
        embedding_id=embedding_result.embedding_id,
        intrinsic_metrics=intrinsic_metrics,
        external_validation_metrics=external_validation_metrics,
        metrics=intrinsic_metrics,
        warnings=warnings,
        notes=notes,
    )


def neighborhood_label_agreement(
    embedding: Any,
    labels: Any,
    n_neighbors: int = 10,
) -> float:
    """Return the fraction of embedded neighbors sharing each sample's label."""
    Y = _as_numeric_array(embedding, name="embedding")
    label_array = np.asarray(labels)
    if label_array.ndim != 1 or len(label_array) != Y.shape[0]:
        raise ValueError("Labels must be one-dimensional and aligned with embedding rows.")
    if Y.shape[0] < 2:
        raise ValueError("At least two samples are required for label agreement.")
    k = min(max(1, n_neighbors), Y.shape[0] - 1)
    neighbors = _neighbor_indices(Y, k)
    agreements = label_array[neighbors] == label_array[:, None]
    return float(np.mean(agreements))


def _external_validation(
    embedding: np.ndarray,
    labels: Any,
    n_neighbors: int,
    warnings: list[str],
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "silhouette_score": None,
        "neighborhood_label_agreement": None,
    }
    label_series = pd.Series(labels).reset_index(drop=True)
    if len(label_series) != embedding.shape[0]:
        warnings.append("External validation skipped because labels are not aligned with embedding rows.")
        return metrics
    if label_series.isna().any():
        warnings.append("External validation skipped because labels contain missing values.")
        return metrics

    class_count = int(label_series.nunique(dropna=False))
    if class_count < 2:
        warnings.append("External validation requires at least two label classes.")
        return metrics
    if class_count >= embedding.shape[0]:
        warnings.append("Silhouette score requires fewer classes than samples.")
    else:
        try:
            metrics["silhouette_score"] = float(
                silhouette_score(embedding, label_series.to_numpy())
            )
        except Exception as exc:
            warnings.append(f"Label silhouette score could not be computed: {exc}")

    try:
        metrics["neighborhood_label_agreement"] = neighborhood_label_agreement(
            embedding,
            label_series.to_numpy(),
            n_neighbors=n_neighbors,
        )
    except Exception as exc:
        warnings.append(f"Neighborhood label agreement could not be computed: {exc}")
    return metrics


def knn_preservation(
    X_original: Any,
    embedding: Any,
    n_neighbors: int = 10,
) -> float:
    """Return mean overlap between original-space and embedding-space kNN sets."""
    X = _as_numeric_array(X_original, name="X_original")
    Y = _as_numeric_array(embedding, name="embedding")
    _validate_shapes(X, Y)
    if X.shape[0] < 2:
        raise ValueError("At least two samples are required for kNN preservation.")

    k = min(max(1, n_neighbors), X.shape[0] - 1)
    original_neighbors = _neighbor_indices(X, k)
    embedding_neighbors = _neighbor_indices(Y, k)
    overlaps = [
        len(set(original).intersection(set(projected))) / k
        for original, projected in zip(original_neighbors, embedding_neighbors, strict=True)
    ]
    return float(np.mean(overlaps))


def distance_correlations(
    X_original: Any,
    embedding: Any,
    sample_size: int = 1000,
    random_seed: int = 42,
) -> tuple[float | None, float | None]:
    """Compare pairwise distances in original and embedded spaces."""
    X = _as_numeric_array(X_original, name="X_original")
    Y = _as_numeric_array(embedding, name="embedding")
    _validate_shapes(X, Y)
    if X.shape[0] < 3:
        raise ValueError("At least three samples are required for distance correlations.")

    X_sample, Y_sample = _sample_rows(X, Y, sample_size, random_seed)
    original_distances = pdist(X_sample)
    embedding_distances = pdist(Y_sample)

    if np.allclose(original_distances, original_distances[0]) or np.allclose(
        embedding_distances, embedding_distances[0]
    ):
        return None, None

    pearson_value = float(pearsonr(original_distances, embedding_distances).statistic)
    spearman_value = float(spearmanr(original_distances, embedding_distances).statistic)
    return _finite_or_none(pearson_value), _finite_or_none(spearman_value)


def _neighbor_indices(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=n_neighbors + 1)
    model.fit(X)
    return model.kneighbors(X, return_distance=False)[:, 1:]


def _sample_rows(
    X: np.ndarray,
    Y: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= sample_size:
        return X, Y
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(X.shape[0], size=sample_size, replace=False)
    return X[indices], Y[indices]


def _as_numeric_array(values: Any, name: str) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        values = values.to_numpy()
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional numeric matrix.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _validate_shapes(X: np.ndarray, embedding: np.ndarray) -> None:
    if X.shape[0] != embedding.shape[0]:
        raise ValueError("Original data and embedding must have the same number of rows.")
    if embedding.shape[1] < 1:
        raise ValueError("Embedding must contain at least one component.")


def _finite_or_none(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return value
