import pandas as pd
from scipy import sparse

from tools.data_inspector import inspect_dataset
from tools.analysis_dataset import AnalysisDataset


def test_inspector_numeric_only_dataset():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})

    profile = inspect_dataset(df, {})

    assert profile.n_samples == 3
    assert profile.n_features == 2
    assert profile.numeric_features == ["x", "y"]
    assert profile.categorical_features == []
    assert profile.missing_fraction == 0


def test_inspector_mixed_types_and_label_config():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "group": ["a", "b", "a"],
            "flag": [True, False, True],
        }
    )

    profile = inspect_dataset(df, {"data": {"label_column": "group"}})

    assert profile.numeric_features == ["x"]
    assert profile.categorical_features == ["group"]
    assert profile.boolean_features == ["flag"]
    assert profile.has_labels is True
    assert profile.label_column == "group"


def test_inspector_missing_constant_sparse_and_duplicate_rows():
    df = pd.DataFrame(
        {
            "x": [0.0, 0.0, None, 1.0],
            "constant": [7, 7, 7, 7],
            "mostly_one": [1, 1, 1, 2],
            "cat": ["a", "a", None, "b"],
        }
    )

    profile = inspect_dataset(df, {"inspection": {"near_constant_threshold": 0.75}})

    assert profile.missing_fraction > 0
    assert "constant" in profile.constant_features
    assert "mostly_one" in profile.near_constant_features
    assert profile.sparsity > 0


def test_inspector_flags_high_dimensional_computational_concerns():
    df = pd.DataFrame(
        {f"x{column}": [column, column + 1] for column in range(12)}
    )

    profile = inspect_dataset(
        df,
        {"inspection": {"high_dimensional_ratio_threshold": 5}},
    )

    assert profile.p_to_n_ratio == 6
    assert profile.computational_concerns
    assert "feature selection" in profile.computational_concerns[0]


def test_sparse_inspection_does_not_densify(monkeypatch):
    matrix = sparse.csr_matrix(
        [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 1.0]]
    )
    dataset = AnalysisDataset(
        X=matrix,
        sample_ids=pd.Index(["s1", "s2", "s3"]),
        feature_ids=pd.Index(["zero", "varying", "binary"]),
        sample_metadata=pd.DataFrame({"batch": [1, 1, 2]}),
        feature_metadata=pd.DataFrame({"name": ["z", "v", "b"]}),
    )

    def forbid_dense(*args, **kwargs):
        raise AssertionError("full sparse matrix was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbid_dense)
    profile = inspect_dataset(dataset, {})

    assert profile.storage_type == "sparse"
    assert profile.sparse_format == "csr"
    assert profile.nonzero_count == 4
    assert profile.density == 4 / 9
    assert profile.nonnegative is True
    assert profile.integer_valued is True
    assert profile.count_like is True
    assert "zero" in profile.constant_features
    assert profile.row_sum_range == (1.0, 4.0)
    assert profile.column_sum_range == (0.0, 6.0)
    assert profile.has_sample_metadata is True
    assert profile.has_feature_metadata is True
