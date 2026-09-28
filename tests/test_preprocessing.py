import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from pydantic import ValidationError

from agent.schemas import PreprocessingAction
from tools.preprocessing import (
    apply_preprocessing_plan,
    normalize_sample_totals,
    select_high_variance_features,
)
from tools.analysis_dataset import AnalysisDataset


def test_imputation_standardization_and_constant_removal():
    df = pd.DataFrame(
        {
            "x": [1.0, None, 3.0],
            "constant": [2.0, 2.0, 2.0],
            "cat": ["a", None, "b"],
        }
    )
    actions = [
        PreprocessingAction(action="impute_numeric", columns=["x"], rationale="Fill numeric."),
        PreprocessingAction(action="impute_categorical", columns=["cat"], rationale="Fill category."),
        PreprocessingAction(action="remove_constant_features", rationale="Drop constants."),
        PreprocessingAction(action="standardize", columns=["x"], rationale="Scale numeric."),
    ]

    result, history = apply_preprocessing_plan(df, actions, random_seed=42)

    assert result["x"].isna().sum() == 0
    assert result["cat"].isna().sum() == 0
    assert "constant" not in result.columns
    assert round(float(result["x"].mean()), 10) == 0
    assert [item["status"] for item in history] == ["success", "success", "success", "success"]


def test_encode_categorical():
    df = pd.DataFrame({"x": [1, 2], "group": ["a", "b"]})
    actions = [
        PreprocessingAction(action="encode_categorical", columns=["group"], rationale="Encode.")
    ]

    result, history = apply_preprocessing_plan(df, actions, random_seed=42)

    assert "group_a" in result.columns
    assert "group_b" in result.columns
    assert history[0]["status"] == "success"


def test_invalid_action_rejected_by_schema():
    with pytest.raises(ValidationError):
        PreprocessingAction(action="made_up", rationale="Invalid.")


def test_missing_column_fails_with_meaningful_error():
    df = pd.DataFrame({"x": [1, 2, 3]})
    actions = [
        PreprocessingAction(action="standardize", columns=["missing"], rationale="Scale.")
    ]

    with pytest.raises(ValueError, match="standardize"):
        apply_preprocessing_plan(df, actions, random_seed=42)


def test_select_high_variance_features_supports_top_k():
    df = pd.DataFrame(
        {
            "low": [1.0, 1.0, 1.0, 2.0],
            "high": [0.0, 10.0, 20.0, 30.0],
            "medium": [0.0, 2.0, 4.0, 6.0],
            "category": ["a", "b", "a", "b"],
        }
    )

    result, metadata = select_high_variance_features(df, top_k=2)

    assert set(result.columns) == {"high", "medium", "category"}
    assert metadata["parameters"]["mode"] == "top_k"
    assert metadata["parameters"]["top_k"] == 2


def test_dense_top_k_preserves_order_values_index_without_fragmentation_warning():
    index = pd.Index(["sample-c", "sample-a", "sample-b"], name="sample_id")
    numeric = {
        f"feature_{position}": np.array([0.0, 1.0, 3.0]) * (position + 1)
        for position in range(240)
    }
    df = pd.DataFrame({"category": ["c", "a", "b"], **numeric}, index=index)
    selected = [f"feature_{position}" for position in range(239, 119, -1)]
    expected = pd.concat([df.loc[:, ["category"]], df.loc[:, selected]], axis=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        result, metadata = select_high_variance_features(df, top_k=120)

    assert not any(
        issubclass(item.category, pd.errors.PerformanceWarning) for item in caught
    )
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_index_equal(result.index, index)
    assert metadata["columns"] == selected


def test_select_high_variance_features_preserves_threshold_mode():
    df = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0],
            "variable": [1.0, 2.0, 3.0],
        }
    )

    result, metadata = select_high_variance_features(df, threshold=0.1)

    assert result.columns.tolist() == ["variable"]
    assert metadata["parameters"]["mode"] == "threshold"
    assert metadata["parameters"]["top_k"] is None


def _sparse_dataset():
    return AnalysisDataset(
        X=sparse.csr_matrix(
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 1.0]]
        ),
        sample_ids=pd.Index(["s1", "s2", "s3"]),
        feature_ids=pd.Index(["constant", "high", "lower"]),
    )


def test_sparse_safe_preprocessing_preserves_sparse_storage():
    actions = [
        PreprocessingAction(action="remove_constant_features", rationale="Remove."),
        PreprocessingAction(
            action="select_high_variance_features",
            parameters={"top_k": 1},
            rationale="Select.",
        ),
        PreprocessingAction(action="log_transform", rationale="Log1p."),
    ]

    result, history = apply_preprocessing_plan(_sparse_dataset(), actions, random_seed=42)

    assert sparse.isspmatrix_csr(result.X)
    assert result.feature_ids.tolist() == ["high"]
    assert result.X.shape == (3, 1)
    assert [item["status"] for item in history] == ["success"] * 3


def test_dense_sample_total_normalization_supports_target_and_zero_rows():
    frame = pd.DataFrame({"a": [1.0, 0.0, 2.0], "b": [3.0, 0.0, 2.0]})

    result, metadata = normalize_sample_totals(frame, target_total=10.0)

    assert np.allclose(result.sum(axis=1), [10.0, 0.0, 10.0])
    assert metadata["parameters"]["target_total"] == 10.0
    assert metadata["zero_total_rows"] == 1
    assert metadata["representation"] == "dense"


def test_sparse_sample_total_normalization_does_not_densify(monkeypatch):
    dataset = AnalysisDataset(
        X=sparse.csr_matrix([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]]),
        sample_ids=pd.Index(["s1", "s2", "s3"]),
        feature_ids=pd.Index(["a", "b"]),
    )

    def reject_densification(*args, **kwargs):
        raise AssertionError("Sparse normalization must not call toarray().")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_densification)
    result, history = apply_preprocessing_plan(
        dataset,
        [PreprocessingAction(
            action="normalize_sample_totals",
            parameters={"target_total": 25.0},
            rationale="Equalize sample totals.",
        )],
        random_seed=42,
    )

    assert sparse.isspmatrix_csr(result.X)
    assert np.allclose(np.asarray(result.X.sum(axis=1)).ravel(), [25.0, 0.0, 25.0])
    assert history[0]["zero_total_rows"] == 1
    assert history[0]["parameters"]["target_total"] == 25.0


def test_variance_selection_uses_transformed_data_in_plan_order():
    frame = pd.DataFrame(
        {
            "library_size_signal": [900.0, 1800.0, 9.0, 18.0],
            "relative_a": [100.0, 200.0, 0.0, 0.0],
            "relative_b": [0.0, 0.0, 1.0, 2.0],
        }
    )
    raw_winner = frame.var(ddof=0).idxmax()
    actions = [
        PreprocessingAction(
            action="normalize_sample_totals",
            parameters={"target_total": 1.0},
            rationale="Normalize totals.",
        ),
        PreprocessingAction(
            action="select_high_variance_features",
            parameters={"top_k": 1},
            rationale="Select after normalization.",
        ),
    ]

    result, history = apply_preprocessing_plan(frame, actions, random_seed=42)

    assert raw_winner == "library_size_signal"
    assert result.columns.tolist() == ["relative_a"]
    assert [item["action"] for item in history] == [
        "normalize_sample_totals", "select_high_variance_features"
    ]


@pytest.mark.parametrize("action", ["standardize", "normalize", "encode_categorical"])
def test_sparse_unsafe_preprocessing_fails_instead_of_densifying(action):
    plan = [PreprocessingAction(action=action, rationale="Unsafe for sparse input.")]

    with pytest.raises(ValueError, match="sparse-safe|sparse input"):
        apply_preprocessing_plan(_sparse_dataset(), plan, random_seed=42)
