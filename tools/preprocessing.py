"""Controlled preprocessing operations for DeEntropy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from agent.schemas import PreprocessingAction
from tools.analysis_dataset import AnalysisDataset


PREPROCESSING_CAPABILITIES = {
    "remove_constant_features": {"dense": True, "sparse": True},
    "impute_numeric": {"dense": True, "sparse": "zero_only"},
    "impute_categorical": {"dense": True, "sparse": False},
    "standardize": {"dense": True, "sparse": False},
    "normalize": {"dense": True, "sparse": False},
    "normalize_sample_totals": {"dense": True, "sparse": True},
    "log_transform": {"dense": True, "sparse": "log1p_nonnegative"},
    "encode_categorical": {"dense": True, "sparse": False},
    "select_high_variance_features": {"dense": True, "sparse": True},
}


def validate_preprocessing_compatibility(
    action: PreprocessingAction,
    storage_type: str,
) -> None:
    support = PREPROCESSING_CAPABILITIES[action.action][storage_type]
    if support is False:
        raise ValueError(
            f"Preprocessing action '{action.action}' is unavailable for {storage_type} input."
        )
    if storage_type == "sparse" and support == "zero_only":
        if action.parameters.get("strategy", "median") != "zero":
            raise ValueError("Sparse numeric imputation supports only strategy='zero'.")
    if storage_type == "sparse" and support == "log1p_nonnegative":
        if float(action.parameters.get("offset", 1.0)) != 1.0:
            raise ValueError("Sparse log transformation supports only offset=1.")


def remove_constant_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_columns = _resolve_columns(df, columns)
    removed = [column for column in target_columns if df[column].nunique(dropna=False) <= 1]
    return df.drop(columns=removed), {
        "action": "remove_constant_features",
        "columns": removed,
        "status": "success",
    }


def impute_numeric(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    strategy: str = "median",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    target_columns = _numeric_columns(result, columns)
    fill_values: dict[str, float] = {}

    for column in target_columns:
        if strategy == "median":
            fill_value = result[column].median()
        elif strategy == "mean":
            fill_value = result[column].mean()
        elif strategy == "zero":
            fill_value = 0
        else:
            raise ValueError(f"Unsupported numeric imputation strategy: {strategy}")
        result[column] = result[column].fillna(fill_value)
        fill_values[column] = float(fill_value) if pd.notna(fill_value) else np.nan

    return result, {
        "action": "impute_numeric",
        "columns": target_columns,
        "parameters": {"strategy": strategy, "fill_values": fill_values},
        "status": "success",
    }


def impute_categorical(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    fill_value: str = "__missing__",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    target_columns = _categorical_columns(result, columns)
    result[target_columns] = result[target_columns].fillna(fill_value)
    return result, {
        "action": "impute_categorical",
        "columns": target_columns,
        "parameters": {"fill_value": fill_value},
        "status": "success",
    }


def standardize(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    target_columns = _numeric_columns(result, columns)
    if target_columns:
        means = result[target_columns].mean()
        stds = result[target_columns].std(ddof=0).replace(0, 1)
        result[target_columns] = (result[target_columns] - means) / stds
    return result, {
        "action": "standardize",
        "columns": target_columns,
        "status": "success",
    }


def normalize(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    target_columns = _numeric_columns(result, columns)
    if target_columns:
        minima = result[target_columns].min()
        ranges = (result[target_columns].max() - minima).replace(0, 1)
        result[target_columns] = (result[target_columns] - minima) / ranges
    return result, {
        "action": "normalize",
        "columns": target_columns,
        "status": "success",
    }


def normalize_sample_totals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    target_total: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scale each sample's selected nonnegative counts to a common row total."""
    if not np.isfinite(target_total) or target_total <= 0:
        raise ValueError("target_total must be a positive finite number.")
    result = df.copy()
    target_columns = _numeric_columns(result, columns)
    values = result[target_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Sample-total normalization requires finite nonnegative values.")
    row_totals = values.sum(axis=1)
    nonzero = row_totals > 0
    scales = np.zeros_like(row_totals, dtype=float)
    scales[nonzero] = target_total / row_totals[nonzero]
    result.loc[:, target_columns] = values * scales[:, None]
    return result, _sample_total_metadata(
        target_columns, target_total, row_totals, representation="dense"
    )


def log_transform(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    offset: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    target_columns = _numeric_columns(result, columns)
    skipped: list[str] = []

    for column in target_columns:
        shifted = result[column] + offset
        if (shifted <= 0).any():
            skipped.append(column)
            continue
        result[column] = np.log(shifted)

    transformed = [column for column in target_columns if column not in skipped]
    return result, {
        "action": "log_transform",
        "columns": transformed,
        "parameters": {"offset": offset, "skipped": skipped},
        "status": "success" if not skipped else "partial",
    }


def encode_categorical(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_columns = _categorical_columns(df, columns)
    result = pd.get_dummies(df, columns=target_columns, drop_first=drop_first, dtype=float)
    added_columns = [column for column in result.columns if column not in df.columns]
    return result, {
        "action": "encode_categorical",
        "columns": target_columns,
        "parameters": {"drop_first": drop_first, "added_columns": added_columns},
        "status": "success",
    }


def select_high_variance_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.0,
    top_k: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    target_columns = _numeric_columns(result, columns)
    if not target_columns:
        return result, {
            "action": "select_high_variance_features",
            "columns": [],
            "parameters": {
                "mode": "top_k" if top_k is not None else "threshold",
                "threshold": threshold,
                "top_k": top_k,
                "removed": [],
            },
            "status": "success",
        }

    variances = result[target_columns].var(ddof=0)
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if top_k > len(target_columns):
            raise ValueError(
                f"top_k={top_k} exceeds the {len(target_columns)} available numeric features."
            )
        column_order = {column: index for index, column in enumerate(target_columns)}
        selected_columns = sorted(
            target_columns,
            key=lambda column: (-float(variances[column]), column_order[column]),
        )[:top_k]
    else:
        selected_columns = [
            column for column in target_columns if float(variances[column]) > threshold
        ]
    removed = [column for column in target_columns if column not in selected_columns]
    result = pd.concat(
        [result.drop(columns=target_columns), df.loc[:, selected_columns]], axis=1
    )
    return result, {
        "action": "select_high_variance_features",
        "columns": selected_columns,
        "parameters": {
            "mode": "top_k" if top_k is not None else "threshold",
            "threshold": threshold,
            "top_k": top_k,
            "removed": removed,
        },
        "status": "success",
    }


def apply_preprocessing_plan(
    df: pd.DataFrame | AnalysisDataset,
    actions: list[PreprocessingAction],
    random_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply validated preprocessing actions in order."""
    del random_seed
    if isinstance(df, AnalysisDataset):
        if df.is_sparse:
            return _apply_sparse_preprocessing_plan(df, actions)
        processed, history = apply_preprocessing_plan(df.X, actions, random_seed=0)
        return df.with_matrix(
            processed,
            feature_ids=processed.columns,
            feature_names=processed.columns,
            feature_metadata=None,
        ), history
    result = df.copy()
    history: list[dict[str, Any]] = []

    for action in actions:
        validate_preprocessing_compatibility(action, "dense")
        try:
            result, metadata = _apply_action(result, action)
        except Exception as exc:
            raise ValueError(
                f"Preprocessing action '{action.action}' failed: {exc}"
            ) from exc
        history.append(metadata)

    return result, history


def _apply_sparse_preprocessing_plan(
    dataset: AnalysisDataset,
    actions: list[PreprocessingAction],
) -> tuple[AnalysisDataset, list[dict[str, Any]]]:
    result = dataset
    history: list[dict[str, Any]] = []
    for action in actions:
        validate_preprocessing_compatibility(action, "sparse")
        try:
            result, metadata = _apply_sparse_action(result, action)
        except Exception as exc:
            raise ValueError(
                f"Preprocessing action '{action.action}' failed for sparse input: {exc}"
            ) from exc
        history.append(metadata)
    return result, history


def _apply_sparse_action(
    dataset: AnalysisDataset,
    action: PreprocessingAction,
) -> tuple[AnalysisDataset, dict[str, Any]]:
    support = PREPROCESSING_CAPABILITIES[action.action]["sparse"]
    if support is False:
        raise ValueError(
            f"'{action.action}' has no sparse-safe implementation and would require densification."
        )
    matrix = dataset.X.tocsr(copy=True)
    target_indices = _sparse_column_indices(dataset, action.columns)
    if action.action == "remove_constant_features":
        variances = _sparse_variances(matrix[:, target_indices])
        remove_indices = set(np.asarray(target_indices)[np.isclose(variances, 0.0)].tolist())
        keep = [index for index in range(matrix.shape[1]) if index not in remove_indices]
        return _slice_sparse_dataset(dataset, keep), {
            "action": action.action,
            "columns": [str(dataset.feature_ids[index]) for index in sorted(remove_indices)],
            "status": "success",
            "representation": "sparse",
        }
    if action.action == "select_high_variance_features":
        variances = _sparse_variances(matrix[:, target_indices])
        top_k = action.parameters.get("top_k")
        threshold = float(action.parameters.get("threshold", 0.0))
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(target_indices):
                raise ValueError("top_k must be a positive integer no larger than the selected feature count.")
            local_keep = np.argsort(-variances, kind="stable")[:top_k]
        else:
            local_keep = np.flatnonzero(variances > threshold)
        selected_target = set(np.asarray(target_indices)[local_keep].tolist())
        untouched = set(range(matrix.shape[1])) - set(target_indices)
        keep = [index for index in range(matrix.shape[1]) if index in selected_target or index in untouched]
        removed = [
            str(dataset.feature_ids[index])
            for index in target_indices
            if index not in selected_target
        ]
        return _slice_sparse_dataset(dataset, keep), {
            "action": action.action,
            "columns": [str(dataset.feature_ids[index]) for index in keep],
            "parameters": {
                "mode": "top_k" if top_k is not None else "threshold",
                "top_k": top_k,
                "threshold": threshold,
                "removed": removed,
            },
            "status": "success",
            "representation": "sparse",
        }
    if action.action == "normalize_sample_totals":
        if len(target_indices) != matrix.shape[1]:
            raise ValueError("Partial-column sparse sample-total normalization is not supported.")
        if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError("Sample-total normalization requires finite nonnegative values.")
        target_total = float(action.parameters.get("target_total", 1.0))
        if not np.isfinite(target_total) or target_total <= 0:
            raise ValueError("target_total must be a positive finite number.")
        row_totals = np.asarray(matrix.sum(axis=1)).ravel()
        scales = np.zeros_like(row_totals, dtype=float)
        nonzero = row_totals > 0
        scales[nonzero] = target_total / row_totals[nonzero]
        transformed = sparse.diags(scales, format="csr") @ matrix
        return dataset.with_matrix(transformed.tocsr()), _sample_total_metadata(
            [str(dataset.feature_ids[index]) for index in target_indices],
            target_total,
            row_totals,
            representation="sparse",
        )
    if action.action == "log_transform":
        if len(target_indices) != matrix.shape[1]:
            raise ValueError("Partial-column sparse log transformation is not supported.")
        offset = float(action.parameters.get("offset", 1.0))
        if offset != 1.0 or np.any(matrix.data < 0):
            raise ValueError("Sparse log transformation supports only nonnegative log1p data with offset=1.")
        transformed = matrix.copy()
        transformed.data = np.log1p(transformed.data)
        return dataset.with_matrix(transformed), {
            "action": action.action,
            "columns": [str(dataset.feature_ids[index]) for index in target_indices],
            "parameters": {"offset": 1.0},
            "status": "success",
            "representation": "sparse",
        }
    if action.action == "impute_numeric":
        if len(target_indices) != matrix.shape[1]:
            raise ValueError("Partial-column sparse imputation is not supported.")
        strategy = action.parameters.get("strategy", "median")
        if strategy != "zero":
            raise ValueError("Sparse numeric imputation supports only strategy='zero'.")
        transformed = matrix.copy()
        transformed.data[~np.isfinite(transformed.data)] = 0
        transformed.eliminate_zeros()
        return dataset.with_matrix(transformed), {
            "action": action.action,
            "columns": [str(dataset.feature_ids[index]) for index in target_indices],
            "parameters": {"strategy": "zero"},
            "status": "success",
            "representation": "sparse",
        }
    raise ValueError(f"'{action.action}' is not implemented for sparse input.")


def _sparse_column_indices(
    dataset: AnalysisDataset,
    columns: list[str] | None,
) -> list[int]:
    if columns is None:
        return list(range(dataset.X.shape[1]))
    positions = {str(feature_id): index for index, feature_id in enumerate(dataset.feature_ids)}
    missing = [column for column in columns if str(column) not in positions]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(map(str, missing))}")
    return [positions[str(column)] for column in columns]


def _sparse_variances(matrix: sparse.spmatrix) -> np.ndarray:
    means = np.asarray(matrix.mean(axis=0)).ravel()
    square_means = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    return np.maximum(square_means - means**2, 0.0)


def _slice_sparse_dataset(dataset: AnalysisDataset, keep: list[int]) -> AnalysisDataset:
    names = dataset.feature_names[keep] if dataset.feature_names is not None else None
    metadata = (
        dataset.feature_metadata.iloc[keep].reset_index(drop=True)
        if dataset.feature_metadata is not None
        else None
    )
    return dataset.with_matrix(
        dataset.X[:, keep].tocsr(),
        feature_ids=dataset.feature_ids[keep],
        feature_names=names,
        feature_metadata=metadata,
    )


def _apply_action(
    df: pd.DataFrame,
    action: PreprocessingAction,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = action.parameters
    if action.action == "remove_constant_features":
        return remove_constant_features(df, action.columns)
    if action.action == "impute_numeric":
        return impute_numeric(df, action.columns, strategy=params.get("strategy", "median"))
    if action.action == "impute_categorical":
        return impute_categorical(
            df,
            action.columns,
            fill_value=params.get("fill_value", "__missing__"),
        )
    if action.action == "standardize":
        return standardize(df, action.columns)
    if action.action == "normalize":
        return normalize(df, action.columns)
    if action.action == "normalize_sample_totals":
        return normalize_sample_totals(
            df,
            action.columns,
            target_total=float(params.get("target_total", 1.0)),
        )
    if action.action == "log_transform":
        return log_transform(df, action.columns, offset=params.get("offset", 1.0))
    if action.action == "encode_categorical":
        return encode_categorical(
            df,
            action.columns,
            drop_first=params.get("drop_first", False),
        )
    if action.action == "select_high_variance_features":
        return select_high_variance_features(
            df,
            action.columns,
            threshold=params.get("threshold", 0.0),
            top_k=params.get("top_k"),
        )
    raise ValueError(f"Unknown preprocessing action: {action.action}")


def _sample_total_metadata(
    columns: list[Any],
    target_total: float,
    original_totals: np.ndarray,
    representation: str,
) -> dict[str, Any]:
    nonzero = original_totals[original_totals > 0]
    return {
        "action": "normalize_sample_totals",
        "columns": [str(column) for column in columns],
        "parameters": {"target_total": float(target_total)},
        "zero_total_rows": int(np.count_nonzero(original_totals == 0)),
        "original_total_range": (
            [float(original_totals.min()), float(original_totals.max())]
            if original_totals.size else None
        ),
        "nonzero_normalized_total_range": (
            [float(target_total), float(target_total)] if nonzero.size else None
        ),
        "status": "success",
        "representation": representation,
    }


def _resolve_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if columns is None:
        return df.columns.tolist()
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")
    return columns


def _numeric_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    target_columns = _resolve_columns(df, columns)
    return [
        column
        for column in target_columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]


def _categorical_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    target_columns = _resolve_columns(df, columns)
    return [
        column
        for column in target_columns
        if not pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]
