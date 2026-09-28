"""Deterministic dataset inspection tools."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from agent.schemas import DatasetProfile
from tools.analysis_dataset import AnalysisDataset


def inspect_dataset(df: pd.DataFrame | AnalysisDataset, config: dict[str, Any] | None = None) -> DatasetProfile:
    """Create a deterministic profile of a dataset without calling an LLM."""
    if isinstance(df, AnalysisDataset):
        if df.is_sparse:
            return _inspect_sparse_dataset(df, config)
        profile = _inspect_dataframe(df.X, config)
        return profile.model_copy(
            update={
                "has_labels": df.labels is not None,
                "label_column": df.label_column if df.labels is not None else None,
                "has_sample_metadata": df.sample_metadata is not None,
                "has_feature_metadata": df.feature_metadata is not None,
            }
        )
    return _inspect_dataframe(df, config)


def _inspect_dataframe(df: pd.DataFrame, config: dict[str, Any] | None = None) -> DatasetProfile:
    config = config or {}
    inspection_config = config.get("inspection", {})
    data_config = config.get("data", {})

    near_constant_threshold = inspection_config.get("near_constant_threshold", 0.99)
    high_cardinality_threshold = inspection_config.get("high_cardinality_threshold", 50)
    small_n_threshold = inspection_config.get("small_n_threshold", 10)
    large_p_threshold = inspection_config.get("large_p_threshold", 1000)
    scaling_ratio_threshold = inspection_config.get("scaling_ratio_threshold", 100)
    high_dimensional_ratio_threshold = inspection_config.get(
        "high_dimensional_ratio_threshold", 5
    )

    n_samples, n_features = df.shape
    warnings: list[str] = []

    numeric_features = [
        column
        for column in df.select_dtypes(include=[np.number]).columns.tolist()
        if not pd.api.types.is_bool_dtype(df[column])
    ]
    boolean_features = [
        column for column in df.columns.tolist() if pd.api.types.is_bool_dtype(df[column])
    ]
    categorical_features = [
        column
        for column in df.columns.tolist()
        if column not in numeric_features and column not in boolean_features
    ]

    missingness_by_column = df.isna().mean().to_dict()
    missing_fraction = float(df.isna().to_numpy().mean()) if n_samples * n_features else 0.0

    numeric_df = df[numeric_features] if numeric_features else pd.DataFrame(index=df.index)
    sparsity = _calculate_sparsity(numeric_df)

    constant_features = [
        column for column in df.columns if df[column].nunique(dropna=False) <= 1
    ]
    near_constant_features = [
        column
        for column in df.columns
        if column not in constant_features
        and _dominant_value_fraction(df[column]) >= near_constant_threshold
    ]

    numeric_summaries = _numeric_summaries(numeric_df)
    potential_scaling_issues = _detect_scaling_issues(
        numeric_summaries,
        scaling_ratio_threshold=scaling_ratio_threshold,
    )
    high_cardinality_categorical_features = [
        column
        for column in categorical_features
        if df[column].nunique(dropna=True) > high_cardinality_threshold
    ]
    computational_concerns: list[str] = []
    p_to_n_ratio = float(n_features / n_samples) if n_samples else math.inf
    if p_to_n_ratio >= high_dimensional_ratio_threshold:
        computational_concerns.append(
            "The high feature-to-sample ratio can make pairwise-distance and manifold methods expensive; consider feature selection."
        )

    if n_samples < small_n_threshold:
        warnings.append(f"Dataset has very small sample size: n={n_samples}.")
    if n_features > large_p_threshold:
        warnings.append(f"Dataset has many features: p={n_features}.")
    if n_samples > 0 and n_features / n_samples > 1:
        warnings.append("Feature count exceeds sample count; p/n is greater than 1.")
    if missing_fraction > 0:
        warnings.append(f"Dataset contains missing values: {missing_fraction:.3f} overall.")
    if not numeric_features:
        warnings.append("No numeric features detected; dimension reduction will require encoding.")
    if constant_features:
        warnings.append(f"Constant features detected: {', '.join(constant_features)}.")
    if near_constant_features:
        warnings.append(f"Near-constant features detected: {', '.join(near_constant_features)}.")
    if potential_scaling_issues:
        warnings.append("Numeric features appear to be on substantially different scales.")
    if high_cardinality_categorical_features:
        warnings.append(
            "High-cardinality categorical features detected: "
            + ", ".join(high_cardinality_categorical_features)
            + "."
        )
    warnings.extend(computational_concerns)

    label_column = data_config.get("label_column")
    has_labels = bool(label_column and label_column in df.columns)

    return DatasetProfile(
        n_samples=n_samples,
        n_features=n_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
        missing_fraction=missing_fraction,
        missingness_by_column={key: float(value) for key, value in missingness_by_column.items()},
        sparsity=sparsity,
        p_to_n_ratio=p_to_n_ratio,
        constant_features=constant_features,
        near_constant_features=near_constant_features,
        duplicate_rows=int(df.duplicated().sum()),
        numeric_summaries=numeric_summaries,
        potential_scaling_issues=potential_scaling_issues,
        high_cardinality_categorical_features=high_cardinality_categorical_features,
        computational_concerns=computational_concerns,
        has_labels=has_labels,
        label_column=label_column if has_labels else None,
        warnings=warnings,
    )


def _inspect_sparse_dataset(
    dataset: AnalysisDataset,
    config: dict[str, Any] | None,
) -> DatasetProfile:
    config = config or {}
    inspection = config.get("inspection", {})
    near_constant_threshold = float(inspection.get("near_constant_threshold", 0.99))
    small_n_threshold = int(inspection.get("small_n_threshold", 10))
    large_p_threshold = int(inspection.get("large_p_threshold", 1000))
    high_dimensional_ratio_threshold = float(
        inspection.get("high_dimensional_ratio_threshold", 5)
    )
    matrix = dataset.X.tocsr(copy=True)
    matrix.eliminate_zeros()
    n_samples, n_features = matrix.shape
    total = n_samples * n_features
    nnz = int(matrix.nnz)
    density = float(nnz / total) if total else 0.0
    data = matrix.data
    finite_mask = np.isfinite(data)
    nonfinite_count = int((~finite_mask).sum())
    finite_data = data[finite_mask]
    nonnegative = bool(np.all(finite_data >= 0)) if finite_data.size else True
    integer_valued = bool(np.all(finite_data == np.floor(finite_data))) if finite_data.size else True

    safe = matrix.copy()
    if nonfinite_count:
        safe.data[~np.isfinite(safe.data)] = 0
        safe.eliminate_zeros()
    column_sums = np.asarray(safe.sum(axis=0)).ravel()
    row_sums = np.asarray(safe.sum(axis=1)).ravel()
    column_square_sums = np.asarray(safe.power(2).sum(axis=0)).ravel()
    means = column_sums / max(n_samples, 1)
    variances = np.maximum(column_square_sums / max(n_samples, 1) - means**2, 0)
    constant_indices = np.flatnonzero(np.isclose(variances, 0.0))
    csc = safe.tocsc()
    nonzero_by_column = np.diff(csc.indptr)
    zero_fraction = 1.0 - nonzero_by_column / max(n_samples, 1)
    near_constant_indices = np.flatnonzero(
        (zero_fraction >= near_constant_threshold)
        & ~np.isin(np.arange(n_features), constant_indices)
    )
    feature_ids = np.asarray(dataset.feature_ids.astype(str))
    constant_features = feature_ids[constant_indices].tolist()
    near_constant_features = feature_ids[near_constant_indices].tolist()
    p_to_n_ratio = float(n_features / n_samples) if n_samples else math.inf
    concerns = [
        "Sparse storage must be preserved; dense conversion may exceed practical memory."
    ]
    if p_to_n_ratio >= high_dimensional_ratio_threshold:
        concerns.append(
            "The high feature-to-sample ratio can make pairwise-distance and manifold methods expensive; consider feature selection."
        )
    warnings: list[str] = []
    if n_samples < small_n_threshold:
        warnings.append(f"Dataset has very small sample size: n={n_samples}.")
    if n_features > large_p_threshold:
        warnings.append(f"Dataset has many features: p={n_features}.")
    if nonfinite_count:
        warnings.append(f"Sparse matrix contains {nonfinite_count} explicitly stored nonfinite values.")
    if constant_features:
        warnings.append(f"Constant features detected: {len(constant_features)}.")
    if near_constant_features:
        warnings.append(f"Near-constant sparse features detected: {len(near_constant_features)}.")
    warnings.extend(concerns)
    positive_std = np.sqrt(variances[variances > 0])
    scaling_issues: list[str] = []
    if positive_std.size >= 2:
        ratio_threshold = float(inspection.get("scaling_ratio_threshold", 100))
        if positive_std.max() / positive_std.min() >= ratio_threshold:
            positive_indices = np.flatnonzero(variances > 0)
            min_index = positive_indices[int(np.argmin(variances[positive_indices]))]
            scaling_issues = [
                str(feature_ids[min_index]),
                str(feature_ids[int(np.argmax(variances))]),
            ]
            warnings.append("Numeric features appear to be on substantially different scales.")

    return DatasetProfile(
        n_samples=n_samples,
        n_features=n_features,
        numeric_features=feature_ids.tolist(),
        categorical_features=[],
        boolean_features=[],
        missing_fraction=float(np.isnan(data).sum() / total) if total else 0.0,
        missingness_by_column={},
        sparsity=1.0 - density,
        p_to_n_ratio=p_to_n_ratio,
        constant_features=constant_features,
        near_constant_features=near_constant_features,
        duplicate_rows=0,
        numeric_summaries={},
        potential_scaling_issues=scaling_issues,
        high_cardinality_categorical_features=[],
        computational_concerns=concerns,
        storage_type="sparse",
        sparse_format=matrix.format,
        nonzero_count=nnz,
        density=density,
        nonnegative=nonnegative,
        integer_valued=integer_valued,
        count_like=nonnegative and integer_valued,
        nonfinite_count=nonfinite_count,
        row_sum_range=_range(row_sums),
        column_sum_range=_range(column_sums),
        has_sample_metadata=dataset.sample_metadata is not None,
        has_feature_metadata=dataset.feature_metadata is not None,
        has_labels=dataset.labels is not None,
        label_column=dataset.label_column if dataset.labels is not None else None,
        warnings=warnings,
    )


def _range(values: np.ndarray) -> tuple[float, float] | None:
    if values.size == 0:
        return None
    return float(values.min()), float(values.max())


def _calculate_sparsity(numeric_df: pd.DataFrame) -> float:
    if numeric_df.empty:
        return 0.0
    values = numeric_df.to_numpy()
    non_missing = ~pd.isna(values)
    denominator = int(non_missing.sum())
    if denominator == 0:
        return 0.0
    return float(((values == 0) & non_missing).sum() / denominator)


def _dominant_value_fraction(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return 0.0
    return float(counts.iloc[0] / len(series))


def _numeric_summaries(numeric_df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    summaries: dict[str, dict[str, float | None]] = {}
    for column in numeric_df.columns:
        series = numeric_df[column]
        summaries[column] = {
            "mean": _finite_or_none(series.mean()),
            "std": _finite_or_none(series.std()),
            "min": _finite_or_none(series.min()),
            "max": _finite_or_none(series.max()),
            "median": _finite_or_none(series.median()),
        }
    return summaries


def _finite_or_none(value: float) -> float | None:
    if pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


def _detect_scaling_issues(
    numeric_summaries: dict[str, dict[str, float | None]],
    scaling_ratio_threshold: float,
) -> list[str]:
    scale_by_column = {
        column: summary["std"]
        for column, summary in numeric_summaries.items()
        if summary["std"] is not None and summary["std"] > 0
    }
    if len(scale_by_column) < 2:
        return []

    min_scale = min(scale_by_column.values())
    max_scale = max(scale_by_column.values())
    if min_scale == 0 or max_scale / min_scale < scaling_ratio_threshold:
        return []

    return [
        column
        for column, scale in scale_by_column.items()
        if scale == min_scale or scale == max_scale
    ]
