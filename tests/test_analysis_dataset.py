import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

from tools.analysis_dataset import AnalysisDataset, load_analysis_dataset


def test_dense_analysis_dataset_preserves_ids_metadata_and_labels():
    dataset = AnalysisDataset(
        X=pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}),
        sample_ids=pd.Index(["s1", "s2"]),
        feature_ids=pd.Index(["f1", "f2"]),
        feature_names=pd.Index(["one", "two"]),
        sample_metadata=pd.DataFrame({"batch": ["a", "b"]}),
        feature_metadata=pd.DataFrame({"kind": ["x", "y"]}),
        labels=pd.Series(["A", "B"]),
    )

    assert dataset.storage_type == "dense"
    assert dataset.X.shape == (2, 2)
    assert dataset.sample_ids.tolist() == ["s1", "s2"]
    assert dataset.labels.tolist() == ["A", "B"]


def test_sparse_analysis_dataset_preserves_sparse_storage():
    matrix = sparse.coo_matrix([[0, 1], [2, 0]])
    dataset = AnalysisDataset(
        X=matrix,
        sample_ids=pd.Index(["s1", "s2"]),
        feature_ids=pd.Index(["f1", "f2"]),
    )

    assert sparse.isspmatrix_csr(dataset.X)
    assert dataset.storage_type == "sparse"


@pytest.mark.parametrize("kind", ["sample", "feature"])
def test_analysis_dataset_rejects_duplicate_stable_ids(kind):
    kwargs = {
        "X": np.ones((2, 2)),
        "sample_ids": pd.Index(["s1", "s2"]),
        "feature_ids": pd.Index(["f1", "f2"]),
    }
    kwargs[f"{kind}_ids"] = pd.Index(["duplicate", "duplicate"])

    with pytest.raises(ValueError, match=f"Duplicate stable {kind} IDs"):
        AnalysisDataset(**kwargs)


def _write_bundle(tmp_path, matrix, orientation):
    matrix_path = tmp_path / "matrix.mtx"
    samples_path = tmp_path / "samples.tsv"
    features_path = tmp_path / "features.tsv"
    mmwrite(matrix_path, matrix)
    samples_path.write_text("s1\tbatch_a\ns2\tbatch_b\n", encoding="utf-8")
    features_path.write_text(
        "id1\tshared\n"
        "id2\tshared\n"
        "id3\tunique\n",
        encoding="utf-8",
    )
    config = {
        "format": "matrix_market",
        "sample_metadata_path": str(samples_path),
        "feature_metadata_path": str(features_path),
        "orientation": orientation,
        "sample_id_column": 0,
        "feature_id_column": 0,
        "feature_name_column": 1,
    }
    return matrix_path, config


@pytest.mark.parametrize(
    ("source", "orientation"),
    [
        (sparse.coo_matrix([[1, 0, 2], [0, 3, 0]]), "samples_by_features"),
        (sparse.coo_matrix([[1, 0], [0, 3], [2, 0]]), "features_by_samples"),
    ],
)
def test_matrix_market_loading_standardizes_orientation_and_aligns_metadata(
    tmp_path, source, orientation
):
    matrix_path, config = _write_bundle(tmp_path, source, orientation)

    dataset = load_analysis_dataset(matrix_path, config)

    assert dataset.X.shape == (2, 3)
    assert sparse.isspmatrix_csr(dataset.X)
    assert dataset.sample_ids.tolist() == ["s1", "s2"]
    assert dataset.feature_ids.tolist() == ["id1", "id2", "id3"]
    assert dataset.feature_names.tolist() == ["shared", "shared", "unique"]
    assert dataset.sample_metadata.iloc[1, 1] == "batch_b"
    assert dataset.matrix_metadata["orientation"] == "samples_by_features"


def test_matrix_market_dimension_mismatch_fails_clearly(tmp_path):
    matrix_path, config = _write_bundle(
        tmp_path,
        sparse.coo_matrix(np.ones((2, 2))),
        "samples_by_features",
    )

    with pytest.raises(ValueError, match="dimensions do not match metadata"):
        load_analysis_dataset(matrix_path, config)


def test_matrix_market_rejects_duplicate_stable_feature_ids(tmp_path):
    matrix_path, config = _write_bundle(
        tmp_path,
        sparse.coo_matrix(np.ones((2, 3))),
        "samples_by_features",
    )
    pd.DataFrame([["dup", "a"], ["dup", "b"], ["id3", "c"]]).to_csv(
        config["feature_metadata_path"], sep="\t", header=False, index=False
    )

    with pytest.raises(ValueError, match="Duplicate stable feature IDs"):
        load_analysis_dataset(matrix_path, config)
