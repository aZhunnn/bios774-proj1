"""Controlled dimension-reduction tools for DeEntropy V1."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import KernelPCA, PCA, SparsePCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding

from agent.method_registry import validate_method_plan
from agent.schemas import EmbeddingResult


SUPPORTED_METHODS = {
    "truncated_svd",
    "pca",
    "sparse_pca",
    "kernel_pca",
    "mds",
    "isomap",
    "lle",
    "spectral_embedding",
    "laplacian_eigenmaps",
    "tsne",
    "umap",
}

UNSUPPORTED_METHODS = {
    "diffusion_maps": "Diffusion Maps requires an additional stable implementation and is not enabled in V1 Phase 2.",
    "gplvm": "GPLVM requires a probabilistic latent-variable modeling stack and is not enabled in V1 Phase 2.",
}


def run_dimension_reduction(
    X: Any,
    method: str,
    n_components: int,
    parameters: dict[str, Any] | None,
    random_seed: int,
) -> EmbeddingResult:
    """Run one approved dimension-reduction method and capture failures safely."""
    start = time.perf_counter()
    normalized_method = _normalize_method(method)
    params = dict(parameters or {})
    embedding_id = _embedding_id(normalized_method, n_components, params, random_seed)

    if normalized_method in UNSUPPORTED_METHODS:
        return _failure_result(
            embedding_id,
            normalized_method,
            params,
            n_components,
            start,
            UNSUPPORTED_METHODS[normalized_method],
        )
    if normalized_method not in SUPPORTED_METHODS:
        return _failure_result(
            embedding_id,
            normalized_method,
            params,
            n_components,
            start,
            f"Unknown dimension-reduction method: {method}",
        )

    try:
        X_array = _as_numeric_matrix(X, allow_sparse=normalized_method == "truncated_svd")
        _validate_input(X_array, n_components)
        validate_method_plan(
            normalized_method,
            n_components,
            params,
            {"n_samples": X_array.shape[0], "n_features": X_array.shape[1]},
        )
        embedding, metadata, effective_params = _fit_transform(
            X_array,
            normalized_method,
            n_components,
            params,
            random_seed,
        )
    except Exception as exc:
        return _failure_result(
            embedding_id,
            normalized_method,
            params,
            n_components,
            start,
            str(exc),
        )

    return EmbeddingResult(
        embedding_id=embedding_id,
        method=normalized_method,
        parameters=effective_params,
        n_components=n_components,
        embedding=embedding,
        runtime_seconds=time.perf_counter() - start,
        success=True,
        error_message=None,
        metadata=metadata,
    )


def _fit_transform(
    X: Any,
    method: str,
    n_components: int,
    parameters: dict[str, Any],
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if method == "truncated_svd":
        params = {"random_state": random_seed, "algorithm": "randomized", **parameters}
        model = TruncatedSVD(n_components=n_components, **params)
        embedding = model.fit_transform(X)
        return embedding, {
            "explained_variance": model.explained_variance_.tolist(),
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "singular_values": model.singular_values_.tolist(),
            "input_was_sparse": bool(sparse.issparse(X)),
            "centered": False,
        }, params

    if method == "pca":
        params = {"random_state": random_seed, **parameters}
        model = PCA(n_components=n_components, **params)
        embedding = model.fit_transform(X)
        return embedding, {
            "explained_variance": model.explained_variance_.tolist(),
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "singular_values": model.singular_values_.tolist(),
        }, params

    if method == "sparse_pca":
        params = {"random_state": random_seed, "alpha": 1.0, **parameters}
        model = SparsePCA(n_components=n_components, **params)
        embedding = model.fit_transform(X)
        return embedding, {"components_shape": list(model.components_.shape)}, params

    if method == "kernel_pca":
        params = {"kernel": "rbf", "fit_inverse_transform": False, **parameters}
        model = KernelPCA(n_components=n_components, **params)
        embedding = model.fit_transform(X)
        metadata = {}
        if hasattr(model, "eigenvalues_"):
            metadata["eigenvalues"] = model.eigenvalues_.tolist()
        return embedding, metadata, params

    if method == "mds":
        params = {"random_state": random_seed, "n_init": 1, "max_iter": 300, **parameters}
        model = MDS(n_components=n_components, dissimilarity="euclidean", **params)
        embedding = model.fit_transform(X)
        return embedding, {"stress": float(model.stress_)}, params

    if method == "isomap":
        params = {"n_neighbors": _default_neighbors(X), **parameters}
        model = Isomap(n_components=n_components, **params)
        embedding = model.fit_transform(X)
        return embedding, {"reconstruction_error": float(model.reconstruction_error())}, params

    if method == "lle":
        params = {"n_neighbors": _default_neighbors(X), "random_state": random_seed, **parameters}
        model = LocallyLinearEmbedding(n_components=n_components, **params)
        embedding = model.fit_transform(X)
        return embedding, {"reconstruction_error": float(model.reconstruction_error_)}, params

    if method in {"spectral_embedding", "laplacian_eigenmaps"}:
        params = {
            "n_neighbors": _default_neighbors(X),
            "random_state": random_seed,
            "affinity": "nearest_neighbors",
            **parameters,
        }
        model = SpectralEmbedding(n_components=n_components, **params)
        embedding = model.fit_transform(X)
        return embedding, {}, params

    if method == "tsne":
        params = {
            "random_state": random_seed,
            "init": "pca",
            "learning_rate": "auto",
            "perplexity": _default_perplexity(X),
            **parameters,
        }
        model = TSNE(n_components=n_components, **params)
        embedding = model.fit_transform(X)
        return embedding, {"kl_divergence": float(model.kl_divergence_)}, params

    if method == "umap":
        try:
            os.environ.setdefault(
                "NUMBA_CACHE_DIR",
                os.path.join(tempfile.gettempdir(), "deentropy_numba_cache"),
            )
            from umap import UMAP
        except Exception as exc:
            raise RuntimeError("UMAP is unavailable; install umap-learn to use method='umap'.") from exc
        params = {
            "n_neighbors": _default_neighbors(X),
            "random_state": random_seed,
            "n_jobs": 1,
            **parameters,
        }
        model = UMAP(n_components=n_components, **params)
        embedding = model.fit_transform(X)
        return embedding, {}, params

    raise ValueError(f"Unknown dimension-reduction method: {method}")


def _as_numeric_matrix(X: Any, allow_sparse: bool = False) -> Any:
    if sparse.issparse(X):
        if allow_sparse:
            matrix = X.astype(float, copy=False)
            if matrix.ndim != 2:
                raise ValueError("Input X must be a two-dimensional numeric matrix.")
            return matrix
        raise ValueError(
            "Sparse input is not supported by the current DeEntropy dimension-reduction execution path."
        )
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError("Input X must be a two-dimensional numeric matrix.")
    return array


def _validate_input(X: Any, n_components: int) -> None:
    if X.shape[0] < 2:
        raise ValueError("At least two samples are required for dimension reduction.")
    if X.shape[1] < 1:
        raise ValueError("At least one feature is required for dimension reduction.")
    if n_components < 1:
        raise ValueError("n_components must be at least 1.")
    values = X.data if sparse.issparse(X) else X
    if not np.isfinite(values).all():
        raise ValueError("Input X contains NaN or infinite values; preprocess before dimension reduction.")


def _default_neighbors(X: np.ndarray) -> int:
    return max(1, min(10, X.shape[0] - 1))


def _default_perplexity(X: np.ndarray) -> float:
    return max(1.0, min(30.0, (X.shape[0] - 1) / 3))


def _normalize_method(method: str) -> str:
    normalized = method.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "spectral": "spectral_embedding",
        "laplacian": "spectral_embedding",
        "laplacian_eigenmap": "spectral_embedding",
        "laplacian_eigenmaps": "spectral_embedding",
        "t_sne": "tsne",
        "t_sne_": "tsne",
    }
    return aliases.get(normalized, normalized)


def _embedding_id(
    method: str,
    n_components: int,
    parameters: dict[str, Any],
    random_seed: int,
) -> str:
    payload = json.dumps(
        {
            "method": method,
            "n_components": n_components,
            "parameters": parameters,
            "random_seed": random_seed,
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{method}_{digest}"


def _failure_result(
    embedding_id: str,
    method: str,
    parameters: dict[str, Any],
    n_components: int,
    start: float,
    error_message: str,
) -> EmbeddingResult:
    return EmbeddingResult(
        embedding_id=embedding_id,
        method=method,
        parameters=parameters,
        n_components=n_components,
        embedding=None,
        runtime_seconds=time.perf_counter() - start,
        success=False,
        error_message=error_message,
        metadata={},
    )
