"""Structured schemas shared by DeEntropy components."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ALLOWED_PREPROCESSING_ACTIONS = {
    "remove_constant_features",
    "impute_numeric",
    "impute_categorical",
    "standardize",
    "normalize",
    "normalize_sample_totals",
    "log_transform",
    "encode_categorical",
    "select_high_variance_features",
}


ALLOWED_DR_METHODS = {
    "pca",
    "truncated_svd",
    "sparse_pca",
    "kernel_pca",
    "mds",
    "isomap",
    "lle",
    "spectral_embedding",
    "tsne",
    "umap",
    "diffusion_maps",
    "gplvm",
}


class DatasetProfile(BaseModel):
    n_samples: int
    n_features: int

    numeric_features: list[str]
    categorical_features: list[str]
    boolean_features: list[str]

    missing_fraction: float
    missingness_by_column: dict[str, float] = Field(default_factory=dict)
    sparsity: float
    p_to_n_ratio: float

    constant_features: list[str]
    near_constant_features: list[str]
    duplicate_rows: int = 0

    numeric_summaries: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    potential_scaling_issues: list[str] = Field(default_factory=list)
    high_cardinality_categorical_features: list[str] = Field(default_factory=list)
    computational_concerns: list[str] = Field(default_factory=list)
    storage_type: Literal["dense", "sparse"] = "dense"
    sparse_format: str | None = None
    nonzero_count: int | None = None
    density: float | None = None
    nonnegative: bool | None = None
    integer_valued: bool | None = None
    count_like: bool | None = None
    nonfinite_count: int = 0
    row_sum_range: tuple[float, float] | None = None
    column_sum_range: tuple[float, float] | None = None
    has_sample_metadata: bool = False
    has_feature_metadata: bool = False

    has_labels: bool
    label_column: str | None

    warnings: list[str]


class PreprocessingAction(BaseModel):
    action: str
    columns: list[str] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str

    @field_validator("action")
    @classmethod
    def action_must_be_allowed(cls, value: str) -> str:
        if value not in ALLOWED_PREPROCESSING_ACTIONS:
            allowed = ", ".join(sorted(ALLOWED_PREPROCESSING_ACTIONS))
            raise ValueError(f"Unknown preprocessing action '{value}'. Allowed: {allowed}")
        return value


class DRMethodPlan(BaseModel):
    method: str
    n_components: int = 2
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str

    @field_validator("method")
    @classmethod
    def method_must_be_allowed(cls, value: str) -> str:
        if value not in ALLOWED_DR_METHODS:
            allowed = ", ".join(sorted(ALLOWED_DR_METHODS))
            raise ValueError(f"Unknown dimension-reduction method '{value}'. Allowed: {allowed}")
        return value


class AnalysisStepPlan(BaseModel):
    step_id: str
    operation: Literal["dimension_reduction"] = "dimension_reduction"
    method: str
    input_ref: str = "preprocessed_X"
    output_ref: str
    output_role: Literal["intermediate", "terminal"] = "terminal"
    n_components: int = 2
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str

    @field_validator("method")
    @classmethod
    def method_must_be_allowed(cls, value: str) -> str:
        if value not in ALLOWED_DR_METHODS:
            allowed = ", ".join(sorted(ALLOWED_DR_METHODS))
            raise ValueError(f"Unknown dimension-reduction method '{value}'. Allowed: {allowed}")
        return value

    @field_validator("step_id", "input_ref", "output_ref")
    @classmethod
    def references_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Step and artifact references must not be blank.")
        return value

    @field_validator("n_components")
    @classmethod
    def components_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_components must be at least 1")
        return value

class AnalysisPlan(BaseModel):
    preprocessing: list[PreprocessingAction] = Field(default_factory=list)
    methods: list[DRMethodPlan] = Field(default_factory=list)
    steps: list[AnalysisStepPlan] = Field(default_factory=list)
    evaluation_metrics: list[str]
    analysis_goal: str
    rationale: str
    alternatives_not_selected: list[str] = Field(default_factory=list)


class ReflectionDecision(BaseModel):
    decision: Literal["continue", "stop"]
    reason: str
    suggested_changes: list[str] = Field(default_factory=list)
    preferred_embedding_id: str | None = None


class EmbeddingResult(BaseModel):
    embedding_id: str
    method: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    n_components: int

    embedding: Any = None

    runtime_seconds: float

    success: bool
    error_message: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class EvaluationResult(BaseModel):
    embedding_id: str
    intrinsic_metrics: dict[str, float | None] = Field(default_factory=dict)
    external_validation_metrics: dict[str, float | None] = Field(default_factory=dict)
    # Compatibility alias for V1 consumers; always mirrors intrinsic_metrics.
    metrics: dict[str, float | None]
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PurposeRecommendation(BaseModel):
    purpose: str
    embedding_id: str
    basis: str


class FinalSynthesis(BaseModel):
    primary_embedding_id: str | None = None
    primary_purpose: str | None = None
    summary: str
    purpose_recommendations: list[PurposeRecommendation] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
