"""Unified dense and sparse analysis dataset representation and loaders."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


_UNSET = object()


@dataclass(frozen=True)
class AnalysisDataset:
    """A samples-by-features matrix with aligned identifiers and metadata."""

    X: Any
    sample_ids: pd.Index
    feature_ids: pd.Index
    feature_names: pd.Index | None = None
    sample_metadata: pd.DataFrame | None = None
    feature_metadata: pd.DataFrame | None = None
    labels: pd.Series | None = None
    input_metadata: dict[str, Any] = field(default_factory=dict)
    matrix_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not hasattr(self.X, "shape") or len(self.X.shape) != 2:
            raise ValueError("AnalysisDataset.X must be a two-dimensional matrix.")
        n_samples, n_features = self.X.shape
        sample_ids = pd.Index(self.sample_ids)
        feature_ids = pd.Index(self.feature_ids)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "feature_ids", feature_ids)
        if len(sample_ids) != n_samples:
            raise ValueError("Sample ID count does not match matrix rows.")
        if len(feature_ids) != n_features:
            raise ValueError("Feature ID count does not match matrix columns.")
        _validate_unique_ids(sample_ids, "sample")
        _validate_unique_ids(feature_ids, "feature")
        if self.feature_names is not None:
            names = pd.Index(self.feature_names)
            if len(names) != n_features:
                raise ValueError("Feature display-name count does not match matrix columns.")
            object.__setattr__(self, "feature_names", names)
        if self.sample_metadata is not None and len(self.sample_metadata) != n_samples:
            raise ValueError("Sample metadata row count does not match matrix rows.")
        if self.feature_metadata is not None and len(self.feature_metadata) != n_features:
            raise ValueError("Feature metadata row count does not match matrix columns.")
        if self.labels is not None:
            labels = pd.Series(self.labels).reset_index(drop=True)
            if len(labels) != n_samples:
                raise ValueError("Label count does not match matrix rows.")
            object.__setattr__(self, "labels", labels)
        if sparse.issparse(self.X) and not sparse.isspmatrix_csr(self.X):
            object.__setattr__(self, "X", self.X.tocsr())

    @property
    def is_sparse(self) -> bool:
        return sparse.issparse(self.X)

    @property
    def storage_type(self) -> str:
        return "sparse" if self.is_sparse else "dense"

    @property
    def features(self) -> Any:
        return self.X

    @property
    def id_column(self) -> str | None:
        return self.input_metadata.get("id_column")

    @property
    def label_column(self) -> str | None:
        return self.input_metadata.get("label_column")

    @property
    def dataframe(self) -> pd.DataFrame:
        if sparse.issparse(self.X):
            raise TypeError("Sparse AnalysisDataset cannot be exposed as a dense dataframe.")
        frame = self.X.copy() if isinstance(self.X, pd.DataFrame) else pd.DataFrame(
            self.X, columns=self.feature_ids
        )
        if self.id_column:
            frame.insert(0, self.id_column, self.sample_ids.to_numpy())
        if self.label_column and self.labels is not None:
            frame[self.label_column] = self.labels.to_numpy()
        return frame

    def with_matrix(
        self,
        X: Any,
        feature_ids: Any | None = None,
        feature_names: Any = _UNSET,
        feature_metadata: Any = _UNSET,
    ) -> "AnalysisDataset":
        return replace(
            self,
            X=X,
            feature_ids=pd.Index(feature_ids if feature_ids is not None else self.feature_ids),
            feature_names=(
                self.feature_names
                if feature_names is _UNSET
                else pd.Index(feature_names) if feature_names is not None else None
            ),
            feature_metadata=(
                self.feature_metadata
                if feature_metadata is _UNSET
                else feature_metadata
            ),
        )


def load_analysis_dataset(
    dataset_path: str | Path,
    data_config: dict[str, Any] | None = None,
) -> AnalysisDataset:
    config = data_config or {}
    path = Path(dataset_path)
    input_format = str(config.get("format") or _infer_format(path)).lower()
    if input_format in {"csv", "tabular"}:
        return _load_csv_dataset(path, config)
    if input_format in {"matrix_market", "mtx"}:
        return _load_matrix_market_dataset(path, config)
    raise ValueError(f"Unsupported input format: {input_format}")


def _load_csv_dataset(path: Path, config: dict[str, Any]) -> AnalysisDataset:
    frame = _read_csv(path)
    id_column = config.get("id_column")
    label_column = config.get("label_column")
    labels_path = config.get("labels_path")
    if id_column:
        _validate_table_id_column(frame, id_column, "feature matrix")
        sample_ids = pd.Index(frame[id_column])
    else:
        sample_ids = pd.Index(frame.index, name="row_index")

    if labels_path:
        if not id_column or not label_column:
            raise ValueError("Separate labels require both data.id_column and data.label_column.")
        labels_frame = _read_csv(Path(labels_path))
        _validate_table_id_column(labels_frame, id_column, "labels file")
        if label_column not in labels_frame:
            raise ValueError(f"Label column '{label_column}' was not found in labels file.")
        missing = sample_ids.difference(pd.Index(labels_frame[id_column]))
        extra = pd.Index(labels_frame[id_column]).difference(sample_ids)
        if len(missing) or len(extra):
            raise ValueError(
                "Feature and label IDs do not match one-to-one "
                f"(missing labels: {len(missing)}, extra labels: {len(extra)})."
            )
        if label_column in frame:
            raise ValueError(f"Label column '{label_column}' already exists in the feature matrix.")
        aligned = labels_frame.set_index(id_column)[label_column].reindex(sample_ids)
        frame = frame.copy()
        frame[label_column] = aligned.to_numpy()

    labels = frame[label_column].reset_index(drop=True) if label_column in frame else None
    metadata_columns = [column for column in (id_column, label_column) if column in frame]
    X = frame.drop(columns=metadata_columns).reset_index(drop=True)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after excluding metadata columns.")
    sample_metadata = frame[metadata_columns].reset_index(drop=True) if metadata_columns else None
    return AnalysisDataset(
        X=X,
        sample_ids=sample_ids,
        feature_ids=pd.Index(X.columns),
        feature_names=pd.Index(X.columns),
        sample_metadata=sample_metadata,
        labels=labels,
        input_metadata={
            "format": "csv",
            "path": str(path),
            "id_column": id_column,
            "label_column": label_column,
            "labels_path": str(labels_path) if labels_path else None,
        },
        matrix_metadata={"orientation": "samples_by_features", "storage_type": "dense"},
    )


def _load_matrix_market_dataset(path: Path, config: dict[str, Any]) -> AnalysisDataset:
    sample_path = config.get("sample_metadata_path")
    feature_path = config.get("feature_metadata_path")
    if not sample_path or not feature_path:
        raise ValueError(
            "Matrix Market input requires data.sample_metadata_path and data.feature_metadata_path."
        )
    try:
        matrix = mmread(path, spmatrix=True)
    except Exception as exc:
        raise ValueError(f"Unable to read Matrix Market input: {path}") from exc
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    orientation = config.get("orientation")
    if orientation not in {"samples_by_features", "features_by_samples"}:
        raise ValueError(
            "Matrix Market orientation must be 'samples_by_features' or 'features_by_samples'."
        )
    if orientation == "features_by_samples":
        matrix = matrix.transpose()
    matrix = matrix.tocsr()

    delimiter = config.get("metadata_delimiter", "\t")
    header = 0 if config.get("metadata_has_header", False) else None
    sample_metadata = _read_metadata(Path(sample_path), delimiter, header, "sample")
    feature_metadata = _read_metadata(Path(feature_path), delimiter, header, "feature")
    sample_id_column = config.get("sample_id_column", 0)
    feature_id_column = config.get("feature_id_column", 0)
    feature_name_column = config.get("feature_name_column")
    label_column = config.get("label_column")
    sample_ids = pd.Index(_metadata_column(sample_metadata, sample_id_column, "sample ID"))
    feature_ids = pd.Index(_metadata_column(feature_metadata, feature_id_column, "feature ID"))
    feature_names = (
        pd.Index(_metadata_column(feature_metadata, feature_name_column, "feature display name"))
        if feature_name_column is not None
        else None
    )
    labels = (
        pd.Series(_metadata_column(sample_metadata, label_column, "label")).reset_index(drop=True)
        if label_column is not None
        else None
    )
    if matrix.shape != (len(sample_ids), len(feature_ids)):
        raise ValueError(
            "Matrix dimensions do not match metadata after orientation: "
            f"matrix={matrix.shape}, samples={len(sample_ids)}, features={len(feature_ids)}."
        )
    return AnalysisDataset(
        X=matrix,
        sample_ids=sample_ids,
        feature_ids=feature_ids,
        feature_names=feature_names,
        sample_metadata=sample_metadata.reset_index(drop=True),
        feature_metadata=feature_metadata.reset_index(drop=True),
        labels=labels,
        input_metadata={
            "format": "matrix_market",
            "path": str(path),
            "sample_metadata_path": str(sample_path),
            "feature_metadata_path": str(feature_path),
            "label_column": str(label_column) if label_column is not None else None,
        },
        matrix_metadata={
            "source_orientation": orientation,
            "orientation": "samples_by_features",
            "storage_type": "sparse",
            "sparse_format": "csr",
        },
    )


def _infer_format(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() == ".mtx":
        return "matrix_market"
    raise ValueError(f"Cannot infer input format from extension: {path.suffix}")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except OSError as exc:
        raise ValueError(f"Unable to read dataset: {path}") from exc


def _read_metadata(
    path: Path,
    delimiter: str,
    header: int | None,
    kind: str,
) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=delimiter, header=header)
    except Exception as exc:
        raise ValueError(f"Unable to read {kind} metadata: {path}") from exc


def _metadata_column(frame: pd.DataFrame, selector: Any, description: str) -> pd.Series:
    try:
        return frame.iloc[:, selector] if isinstance(selector, int) else frame[selector]
    except (IndexError, KeyError) as exc:
        raise ValueError(f"Configured {description} column {selector!r} was not found.") from exc


def _validate_unique_ids(values: pd.Index, kind: str) -> None:
    if values.isna().any():
        raise ValueError(f"Stable {kind} IDs contain missing values.")
    duplicates = values[values.duplicated()].unique()
    if len(duplicates):
        raise ValueError(
            f"Duplicate stable {kind} IDs found: " + ", ".join(map(str, duplicates[:5]))
        )


def _validate_table_id_column(frame: pd.DataFrame, column: str, source: str) -> None:
    if column not in frame:
        raise ValueError(f"ID column '{column}' was not found in {source}.")
    try:
        _validate_unique_ids(pd.Index(frame[column]), source)
    except ValueError as exc:
        raise ValueError(f"Duplicate IDs found in {source}: {exc}") from exc
