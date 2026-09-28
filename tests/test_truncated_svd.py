import numpy as np
from scipy import sparse

from tools.dimension_reduction import run_dimension_reduction


def test_truncated_svd_accepts_sparse_input_without_densifying_root(monkeypatch):
    X = sparse.random(30, 20, density=0.15, random_state=42, format="csr")

    def reject_densification(*args, **kwargs):
        raise AssertionError("Sparse root must not be converted with toarray().")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_densification)
    result = run_dimension_reduction(X, "truncated_svd", 5, {}, random_seed=42)

    assert result.success is True
    assert np.asarray(result.embedding).shape == (30, 5)
    assert result.metadata["input_was_sparse"] is True
    assert result.metadata["centered"] is False


def test_truncated_svd_is_deterministic_for_configured_seed():
    X = sparse.random(25, 15, density=0.25, random_state=9, format="csr")
    first = run_dimension_reduction(X, "truncated_svd", 4, {}, random_seed=17)
    second = run_dimension_reduction(X, "truncated_svd", 4, {}, random_seed=17)

    assert first.success and second.success
    assert np.allclose(first.embedding, second.embedding)
