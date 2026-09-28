"""Structured reflection component for DeEntropy Phase 3."""

from __future__ import annotations

from typing import Any

from agent.llm_client import LLMClient
from agent.method_registry import assess_method_feasibility, method_capability_context
from agent.prompts import build_reflector_prompt
from agent.schemas import (
    ALLOWED_PREPROCESSING_ACTIONS,
    FinalSynthesis,
    PurposeRecommendation,
    ReflectionDecision,
)
from agent.state import AgentState


VALID_SUGGESTION_TOKENS = {
    "remove_constant_features",
    "impute_numeric",
    "impute_categorical",
    "standardize",
    "normalize",
    "log_transform",
    "encode_categorical",
    "select_high_variance_features",
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
    "n_neighbors",
    "perplexity",
    "min_dist",
}


def reflect(state: AgentState, llm_client: LLMClient) -> ReflectionDecision:
    """Return a validated continue/stop decision."""
    state.record_final_synthesis(synthesize_completed_results(state))
    context = build_reflector_context(state)
    deterministic_stop = _deterministic_stop(context)
    if deterministic_stop is not None:
        state.record_reflection(deterministic_stop)
        state.finished = deterministic_stop.decision == "stop"
        return deterministic_stop

    prompt = build_reflector_prompt(context)
    decision = llm_client.generate_structured(prompt, ReflectionDecision)
    _validate_suggestions(decision)
    state.record_reflection(decision)
    if decision.decision == "stop":
        state.finished = True
    return decision


def build_reflector_context(state: AgentState) -> dict[str, Any]:
    best_evaluation = _select_best_evaluation(state.evaluation_history)
    profile = state.dataset_profile.model_dump() if state.dataset_profile else {}
    capabilities = method_capability_context()
    return {
        "dataset_path": state.dataset_path,
        "dataset_profile": profile if state.dataset_profile else None,
        "analysis_history": state.analysis_history,
        "terminal_experiment_history": [
            item
            for item in state.analysis_history
            if item.get("output_role", "terminal") == "terminal"
        ],
        "evaluation_history": state.evaluation_history,
        "reflection_history": state.reflection_history,
        "warnings": state.warnings,
        "errors": state.errors,
        "current_iteration": state.iteration,
        "max_iterations": state.max_iterations,
        "iteration_limit_reached": state.iteration >= state.max_iterations,
        "max_experiments": state.max_experiments,
        "experiment_count": state.experiments_used(),
        "remaining_experiment_budget": max(
            0,
            state.max_experiments - state.experiments_used(),
        ),
        "experiment_budget_exhausted": state.experiment_budget_exhausted(),
        "analysis_step_count": state.analysis_steps_used(),
        "max_analysis_steps": state.max_analysis_steps,
        "analysis_step_limit_exhausted": state.analysis_step_limit_exhausted(),
        "best_embedding_id": state.best_embedding_id,
        "best_evaluation": best_evaluation,
        "available_artifacts": [
            item for item in state.artifact_history if item.get("success", True)
        ],
        "available_preprocessing_tools": sorted(ALLOWED_PREPROCESSING_ACTIONS),
        "method_capabilities": capabilities,
        "method_feasibility": {
            method: assess_method_feasibility(method, profile)
            for method in capabilities
        },
    }


def _deterministic_stop(context: dict[str, Any]) -> ReflectionDecision | None:
    if context["iteration_limit_reached"]:
        return ReflectionDecision(
            decision="stop",
            reason="Maximum iteration budget has been reached.",
            preferred_embedding_id=context.get("best_embedding_id"),
        )
    if context["experiment_budget_exhausted"]:
        return ReflectionDecision(
            decision="stop",
            reason="Experiment budget is exhausted.",
            preferred_embedding_id=context.get("best_embedding_id"),
        )
    if context["analysis_step_limit_exhausted"]:
        return ReflectionDecision(
            decision="stop",
            reason="Maximum analysis-step safety limit has been reached.",
            preferred_embedding_id=context.get("best_embedding_id"),
        )
    return None


def _select_best_evaluation(
    evaluation_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return _best_by_metrics(
        evaluation_history,
        "intrinsic_metrics",
        ("knn_preservation", "trustworthiness"),
    )


def synthesize_completed_results(state: AgentState) -> FinalSynthesis:
    """Compare completed embeddings by purpose without averaging unlike metrics."""
    evaluations = state.evaluation_history
    if not evaluations:
        return FinalSynthesis(
            summary="No successful embedding evaluation is available for recommendation."
        )

    local = _best_by_metrics(
        evaluations,
        "intrinsic_metrics",
        ("knn_preservation", "trustworthiness"),
    )
    global_structure = _best_by_metrics(
        evaluations,
        "intrinsic_metrics",
        ("distance_spearman_correlation", "distance_pearson_correlation"),
    )
    external = _best_by_metrics(
        evaluations,
        "external_validation_metrics",
        ("silhouette_score", "neighborhood_label_agreement"),
    )

    recommendations: list[PurposeRecommendation] = []
    if local:
        recommendations.append(
            PurposeRecommendation(
                purpose="local-neighborhood exploration",
                embedding_id=local["embedding_id"],
                basis=(
                    "Compared kNN preservation first, then trustworthiness; both assess "
                    "local-neighborhood preservation without combining them into one score."
                    + _recorded_method_context(local)
                ),
            )
        )
    if global_structure:
        recommendations.append(
            PurposeRecommendation(
                purpose="broader pairwise-structure reference",
                embedding_id=global_structure["embedding_id"],
                basis=(
                    "Compared Spearman distance correlation first, then Pearson distance "
                    "correlation; these summarize broader pairwise-distance preservation."
                    + _recorded_method_context(global_structure)
                ),
            )
        )
    if external:
        recommendations.append(
            PurposeRecommendation(
                purpose="post-hoc alignment with known labels",
                embedding_id=external["embedding_id"],
                basis=(
                    "Compared label silhouette first, then neighborhood label agreement. "
                    "Labels were used only for external validation."
                    + _recorded_method_context(external)
                ),
            )
        )

    primary = local or global_structure or external
    primary_id = primary.get("embedding_id") if primary else None
    primary_purpose = recommendations[0].purpose if recommendations else None
    tradeoffs: list[str] = []
    if local and global_structure and local["embedding_id"] != global_structure["embedding_id"]:
        tradeoffs.append(
            f"`{local['embedding_id']}` is preferred for local-neighborhood exploration, "
            f"while `{global_structure['embedding_id']}` is complementary for broader "
            "pairwise-structure preservation."
        )
    if external and primary_id and external["embedding_id"] != primary_id:
        tradeoffs.append(
            f"`{external['embedding_id']}` shows the strongest post-hoc alignment with known "
            "labels, which is separate from intrinsic embedding quality."
        )

    summary = (
        f"Use `{primary_id}` for {primary_purpose}."
        if primary_id and primary_purpose
        else "No embedding had sufficient computed metrics for a purpose-specific recommendation."
    )
    return FinalSynthesis(
        primary_embedding_id=primary_id,
        primary_purpose=primary_purpose,
        summary=summary,
        purpose_recommendations=recommendations,
        tradeoffs=tradeoffs,
    )


def _best_by_metrics(
    evaluations: list[dict[str, Any]],
    metric_group: str,
    ordered_metrics: tuple[str, ...],
) -> dict[str, Any] | None:
    candidates = []
    for item in evaluations:
        metrics = _metric_group(item, metric_group)
        if any(metrics.get(name) is not None for name in ordered_metrics):
            candidates.append(item)
    if not candidates:
        return None

    def rank(item: dict[str, Any]) -> tuple[float, ...]:
        metrics = _metric_group(item, metric_group)
        return tuple(
            float(metrics[name]) if metrics.get(name) is not None else float("-inf")
            for name in ordered_metrics
        )

    return max(candidates, key=rank)


def _metric_group(item: dict[str, Any], metric_group: str) -> dict[str, Any]:
    if metric_group == "intrinsic_metrics":
        return item.get("intrinsic_metrics") or item.get("metrics") or {}
    return item.get(metric_group) or {}


def _recorded_method_context(item: dict[str, Any]) -> str:
    notes = [str(note) for note in item.get("notes", []) if note]
    if not notes:
        return ""
    return " Recorded method context: " + " ".join(notes)


def _validate_suggestions(decision: ReflectionDecision) -> None:
    invalid = []
    for suggestion in decision.suggested_changes:
        if not any(token in suggestion for token in VALID_SUGGESTION_TOKENS):
            invalid.append(suggestion)
    if invalid:
        raise ValueError(
            "Reflector suggested invalid next actions: " + "; ".join(invalid)
        )
