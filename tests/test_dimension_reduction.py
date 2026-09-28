import numpy as np
from scipy import sparse

from tools.dimension_reduction import run_dimension_reduction


def make_matrix(n_samples=30, n_features=5):
    rng = np.random.default_rng(123)
    return rng.normal(size=(n_samples, n_features))


def test_pca_output_dimensions_and_metadata():
    result = run_dimension_reduction(
        make_matrix(),
        method="pca",
        n_components=2,
        parameters={},
        random_seed=42,
    )

    assert result.success is True
    assert result.method == "pca"
    assert result.embedding.shape == (30, 2)
    assert "explained_variance_ratio" in result.metadata


def test_umap_output_dimensions_when_available():
    result = run_dimension_reduction(
        make_matrix(),
        method="umap",
        n_components=2,
        parameters={"n_neighbors": 5, "n_epochs": 20},
        random_seed=42,
    )

    assert result.success is True
    assert result.method == "umap"
    assert result.embedding.shape == (30, 2)


def test_additional_stable_methods_return_expected_shape():
    X = make_matrix(n_samples=24, n_features=4)
    method_params = {
        "sparse_pca": {"max_iter": 50},
        "kernel_pca": {"kernel": "linear"},
        "mds": {"max_iter": 50, "eps": 1e-3},
        "isomap": {"n_neighbors": 5},
        "lle": {"n_neighbors": 6, "method": "standard"},
        "spectral_embedding": {"n_neighbors": 5},
        "tsne": {"perplexity": 5, "max_iter": 250},
    }

    for method, params in method_params.items():
        result = run_dimension_reduction(
            X,
            method=method,
            n_components=2,
            parameters=params,
            random_seed=42,
        )

        assert result.success is True, result.error_message
        assert result.embedding.shape == (24, 2)


def test_unsupported_methods_fail_clearly():
    result = run_dimension_reduction(
        make_matrix(),
        method="diffusion_maps",
        n_components=2,
        parameters={},
        random_seed=42,
    )

    assert result.success is False
    assert "not enabled" in result.error_message
    assert result.embedding is None


def test_invalid_method_and_bad_input_are_captured_as_failures():
    invalid_method = run_dimension_reduction(
        make_matrix(),
        method="not_a_method",
        n_components=2,
        parameters={},
        random_seed=42,
    )
    bad_input = run_dimension_reduction(
        np.array([[1.0, np.nan], [2.0, 3.0]]),
        method="pca",
        n_components=2,
        parameters={},
        random_seed=42,
    )

    assert invalid_method.success is False
    assert "Unknown dimension-reduction method" in invalid_method.error_message
    assert bad_input.success is False
    assert "NaN or infinite" in bad_input.error_message


def test_sparse_input_fails_clearly_without_densification():
    result = run_dimension_reduction(
        sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]]),
        method="pca",
        n_components=1,
        parameters={},
        random_seed=42,
    )

    assert result.success is False
    assert "Sparse input is not supported" in result.error_message
