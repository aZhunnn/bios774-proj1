import pytest

from agent.method_registry import (
    METHOD_CAPABILITIES,
    assess_method_feasibility,
    executable_method_names,
    validate_method_plan,
)


def profile(n_samples=100, n_features=10):
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "p_to_n_ratio": n_features / n_samples,
    }


def test_registry_describes_roles_input_support_and_executability():
    assert set(METHOD_CAPABILITIES["pca"].supported_roles) == {"terminal", "intermediate"}
    assert METHOD_CAPABILITIES["tsne"].supported_roles == ("terminal",)
    assert METHOD_CAPABILITIES["umap"].dense_input is True
    assert METHOD_CAPABILITIES["umap"].sparse_input is False
    assert METHOD_CAPABILITIES["diffusion_maps"].executable is False
    assert METHOD_CAPABILITIES["gplvm"].executable is False


def test_unsupported_methods_are_never_operational():
    executable = set(executable_method_names())

    assert "diffusion_maps" not in executable
    assert "gplvm" not in executable
    assert assess_method_feasibility("diffusion_maps", profile())["feasible"] is False
    assert assess_method_feasibility("gplvm", profile())["feasible"] is False


def test_quadratic_methods_are_infeasible_for_large_sample_counts():
    large = profile(n_samples=3000, n_features=20)

    assert assess_method_feasibility("mds", large)["feasible"] is False
    assert assess_method_feasibility("kernel_pca", large)["feasible"] is False
    assert assess_method_feasibility("pca", large)["feasible"] is True


def test_sparse_representation_is_used_by_method_feasibility():
    sparse_profile = {**profile(), "storage_type": "sparse"}

    for method in executable_method_names():
        result = assess_method_feasibility(method, sparse_profile)
        if method == "truncated_svd":
            assert result["feasible"] is True
        else:
            assert result["feasible"] is False
            assert any("sparse input" in reason for reason in result["reasons"])


def test_method_parameter_validation_rejects_unknown_and_invalid_kernel_values():
    with pytest.raises(ValueError, match="Unsupported parameters"):
        validate_method_plan("pca", 2, {"made_up": 1}, profile())
    with pytest.raises(ValueError, match="allowed"):
        validate_method_plan("kernel_pca", 2, {"kernel": "unknown"}, profile())
    with pytest.raises(ValueError, match="at least"):
        validate_method_plan("kernel_pca", 2, {"gamma": -1}, profile())


def test_component_count_is_bounded_by_data_shape():
    with pytest.raises(ValueError, match="n_components"):
        validate_method_plan("pca", 11, {}, profile(n_samples=20, n_features=10))
