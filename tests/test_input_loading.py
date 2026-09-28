import pandas as pd
import pytest

from main import load_analysis_input
from tools.dimension_reduction import run_dimension_reduction


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_separate_labels_are_joined_by_id_in_feature_order(tmp_path):
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"
    _write_csv(
        features_path,
        {"sample_id": ["s2", "s1", "s3"], "x": [2.0, 1.0, 3.0]},
    )
    _write_csv(
        labels_path,
        {"sample_id": ["s1", "s3", "s2"], "class": ["A", "C", "B"]},
    )

    loaded = load_analysis_input(
        features_path,
        {
            "labels_path": str(labels_path),
            "id_column": "sample_id",
            "label_column": "class",
        },
    )

    assert loaded.sample_ids.tolist() == ["s2", "s1", "s3"]
    assert loaded.labels.tolist() == ["B", "A", "C"]
    assert loaded.features.columns.tolist() == ["x"]
    assert loaded.dataframe["class"].tolist() == ["B", "A", "C"]


def test_mismatched_separate_label_ids_fail_clearly(tmp_path):
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"
    _write_csv(features_path, {"id": [1, 2], "x": [3.0, 4.0]})
    _write_csv(labels_path, {"id": [1, 3], "label": ["a", "b"]})

    with pytest.raises(ValueError, match="IDs do not match one-to-one"):
        load_analysis_input(
            features_path,
            {
                "labels_path": str(labels_path),
                "id_column": "id",
                "label_column": "label",
            },
        )


@pytest.mark.parametrize("duplicate_source", ["features", "labels"])
def test_duplicate_ids_fail_clearly(tmp_path, duplicate_source):
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"
    feature_ids = [1, 1] if duplicate_source == "features" else [1, 2]
    label_ids = [1, 1] if duplicate_source == "labels" else [1, 2]
    _write_csv(features_path, {"id": feature_ids, "x": [3.0, 4.0]})
    _write_csv(labels_path, {"id": label_ids, "label": ["a", "b"]})

    with pytest.raises(ValueError, match="Duplicate IDs"):
        load_analysis_input(
            features_path,
            {
                "labels_path": str(labels_path),
                "id_column": "id",
                "label_column": "label",
            },
        )


def test_ids_and_labels_are_not_dimension_reduction_features(tmp_path):
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"
    _write_csv(
        features_path,
        {"id": [10, 20, 30, 40], "x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 1.0, 3.0, 2.0]},
    )
    _write_csv(
        labels_path,
        {"id": [40, 20, 10, 30], "label": ["d", "b", "a", "c"]},
    )

    loaded = load_analysis_input(
        features_path,
        {
            "labels_path": str(labels_path),
            "id_column": "id",
            "label_column": "label",
        },
    )
    result = run_dimension_reduction(
        loaded.features,
        method="pca",
        n_components=2,
        parameters={},
        random_seed=42,
    )

    assert loaded.features.columns.tolist() == ["x", "y"]
    assert loaded.labels.tolist() == ["a", "b", "c", "d"]
    assert result.success is True
    assert result.embedding.shape == (4, 2)
