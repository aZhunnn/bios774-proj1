"""Structured planning component for DeEntropy Phase 3."""

from __future__ import annotations

import json
from typing import Any

from agent.llm_client import LLMClient
from agent.method_registry import (
    assess_method_feasibility,
    executable_method_names,
    method_capability_context,
    validate_method_plan,
)
from agent.prompts import build_planner_prompt
from agent.schemas import (
    ALLOWED_PREPROCESSING_ACTIONS,
    AnalysisPlan,
    AnalysisStepPlan,
    DRMethodPlan,
)
from agent.state import AgentState
from tools.preprocessing import (
    PREPROCESSING_CAPABILITIES,
    validate_preprocessing_compatibility,
)


AVAILABLE_PREPROCESSING_TOOLS = sorted(ALLOWED_PREPROCESSING_ACTIONS)
AVAILABLE_DIMENSION_REDUCTION_TOOLS = executable_method_names()
DEFAULT_EVALUATION_METRICS = [
    "trustworthiness",
    "knn_preservation",
    "distance_pearson_correlation",
    "distance_spearman_correlation",
]
ROOT_ARTIFACT_REF = "preprocessed_X"


def create_plan(state: AgentState, llm_client: LLMClient) -> AnalysisPlan:
    """Ask an LLM client for the next validated analysis plan."""
    context = build_planner_context(state)
    prompt = build_planner_prompt(context)
    plan = llm_client.generate_structured(prompt, AnalysisPlan)
    plan = _enforce_plan_guardrails(plan, context)
    state.current_plan = plan
    return plan


def build_planner_context(state: AgentState) -> dict[str, Any]:
    """Serialize only the compact information needed for planning."""
    attempted_methods = [
        str(item.get("method"))
        for item in state.analysis_history
        if item.get("method") is not None
    ]
    completed_terminal_experiments = [
        {
            "method": item.get("method"),
            "parameters": item.get("planned_parameters", item.get("parameters", {})),
            "n_components": item.get("n_components"),
            "input_ref": item.get("input_ref"),
            "output_ref": item.get("output_ref"),
            "success": item.get("success"),
        }
        for item in state.analysis_history
        if item.get("output_role", "terminal") == "terminal"
        and item.get("execution_status") != "blocked_dependency"
    ]
    available_artifacts = [
        item for item in state.artifact_history if item.get("success", True)
    ]
    remaining_experiment_budget = max(
        0,
        state.max_experiments - state.experiments_used(),
    )
    profile = state.dataset_profile.model_dump() if state.dataset_profile else {}
    method_feasibility = {
        method: assess_method_feasibility(method, profile)
        for method in method_capability_context()
    }
    return {
        "dataset_path": state.dataset_path,
        "dataset_profile": (
            profile if state.dataset_profile else None
        ),
        "preprocessing_history": state.preprocessing_history,
        "attempted_methods": attempted_methods,
        "completed_terminal_experiments": completed_terminal_experiments,
        "available_artifacts": available_artifacts,
        "latest_reflector_feedback": (
            state.reflection_history[-1] if state.reflection_history else None
        ),
        "previous_plans": state.plan_history,
        "analysis_history": state.analysis_history,
        "evaluation_history": state.evaluation_history,
        "reflection_history": state.reflection_history,
        "current_iteration": state.iteration,
        "max_iterations": state.max_iterations,
        "remaining_iterations": max(0, state.max_iterations - state.iteration),
        "max_experiments": state.max_experiments,
        "remaining_experiment_budget": remaining_experiment_budget,
        "max_analysis_steps": state.max_analysis_steps,
        "remaining_analysis_steps": max(
            0, state.max_analysis_steps - state.analysis_steps_used()
        ),
        "available_preprocessing_tools": AVAILABLE_PREPROCESSING_TOOLS,
        "preprocessing_capabilities": PREPROCESSING_CAPABILITIES,
        "available_dimension_reduction_tools": AVAILABLE_DIMENSION_REDUCTION_TOOLS,
        "method_capabilities": method_capability_context(),
        "method_feasibility": method_feasibility,
        "available_evaluation_metrics": DEFAULT_EVALUATION_METRICS,
    }


def _enforce_plan_guardrails(
    plan: AnalysisPlan,
    context: dict[str, Any],
) -> AnalysisPlan:
    remaining_budget = int(context["remaining_experiment_budget"])
    available_methods = set(context["available_dimension_reduction_tools"])
    available_actions = set(context["available_preprocessing_tools"])
    profile = context.get("dataset_profile") or {}
    storage_type = profile.get("storage_type", "dense")

    for action in plan.preprocessing:
        if action.action not in available_actions:
            raise ValueError(f"Planner selected unavailable preprocessing action: {action.action}")
        validate_preprocessing_compatibility(action, storage_type)
    validate_preprocessing_order(plan.preprocessing)
    steps = list(plan.steps)
    if not steps:
        steps = [
            AnalysisStepPlan(
                step_id=f"{method.method}_{index}",
                method=method.method,
                input_ref=ROOT_ARTIFACT_REF,
                output_ref=f"artifact_{method.method}_{index}",
                output_role="terminal",
                n_components=method.n_components,
                parameters=method.parameters,
                rationale=method.rationale,
            )
            for index, method in enumerate(plan.methods, start=1)
        ]

    external_profiles = {
        str(item["artifact_id"]): item["profile"]
        for item in context.get("available_artifacts", [])
        if item.get("artifact_id") and item.get("profile")
    }
    ordered_steps = validate_and_order_steps(
        steps, profile, available_methods, external_profiles
    )
    completed_signatures = {
        _terminal_signature(
            item.get("method"),
            item.get("input_ref"),
            item.get("n_components"),
            item.get("parameters") or {},
        )
        for item in context.get("completed_terminal_experiments", [])
    }
    terminal_steps = [
        step
        for step in ordered_steps
        if step.output_role == "terminal"
        and _terminal_signature(
            step.method, step.input_ref, step.n_components, step.parameters
        ) not in completed_signatures
    ]
    selected_terminals = terminal_steps[:remaining_budget]
    required_refs = {step.output_ref for step in selected_terminals}
    required_steps: set[str] = set()
    by_output = {step.output_ref: step for step in ordered_steps}
    pending_refs = list(required_refs)
    while pending_refs:
        output_ref = pending_refs.pop()
        producer = by_output.get(output_ref)
        if producer is None:
            # This dependency is a previously materialized artifact supplied
            # to the planner, rather than a step produced by the current plan.
            continue
        if producer.step_id in required_steps:
            continue
        required_steps.add(producer.step_id)
        if producer.input_ref != ROOT_ARTIFACT_REF:
            pending_refs.append(producer.input_ref)

    budget_safe_steps = [
        step for step in ordered_steps if step.step_id in required_steps
    ]
    remaining_steps = int(context.get("remaining_analysis_steps", len(budget_safe_steps)))
    if len(budget_safe_steps) > remaining_steps:
        raise ValueError(
            f"Analysis plan requires {len(budget_safe_steps)} steps but only "
            f"{remaining_steps} analysis-step slots remain."
        )

    methods = [
        DRMethodPlan(
            method=step.method,
            n_components=step.n_components,
            parameters=step.parameters,
            rationale=step.rationale,
        )
        for step in budget_safe_steps
    ]
    return plan.model_copy(update={"steps": budget_safe_steps, "methods": methods})


def _terminal_signature(
    method: Any,
    input_ref: Any,
    n_components: Any,
    parameters: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "method": method,
            "input_ref": input_ref,
            "n_components": n_components,
            "parameters": parameters,
        },
        sort_keys=True,
        default=str,
    )


def validate_preprocessing_order(actions: list[Any]) -> None:
    """Reject plans that compute feature variance before value transformations."""
    names = [action.action for action in actions]
    positions = {name: index for index, name in enumerate(names)}
    normalization = positions.get("normalize_sample_totals")
    log_transform = positions.get("log_transform")
    selection = positions.get("select_high_variance_features")
    if normalization is not None and log_transform is not None and normalization > log_transform:
        raise ValueError("normalize_sample_totals must precede log_transform.")
    for transformation in (normalization, log_transform):
        if transformation is not None and selection is not None and transformation > selection:
            raise ValueError(
                "select_high_variance_features must follow count-value transformations."
            )


def validate_and_order_steps(
    steps: list[AnalysisStepPlan],
    root_profile: dict[str, Any],
    available_methods: set[str] | None = None,
    available_artifact_profiles: dict[str, dict[str, Any]] | None = None,
) -> list[AnalysisStepPlan]:
    """Validate artifact references and return a stable topological ordering."""
    available = available_methods or set(AVAILABLE_DIMENSION_REDUCTION_TOOLS)
    step_ids: set[str] = set()
    output_to_step: dict[str, AnalysisStepPlan] = {}
    for step in steps:
        if step.step_id in step_ids:
            raise ValueError(f"Duplicate analysis step_id: {step.step_id}")
        if step.output_ref == ROOT_ARTIFACT_REF or step.output_ref in output_to_step:
            raise ValueError(f"Duplicate or reserved output artifact reference: {step.output_ref}")
        if step.method not in available:
            raise ValueError(f"Planner selected unavailable DR method: {step.method}")
        step_ids.add(step.step_id)
        output_to_step[step.output_ref] = step

    external_profiles = available_artifact_profiles or {}
    for step in steps:
        if (
            step.input_ref != ROOT_ARTIFACT_REF
            and step.input_ref not in output_to_step
            and step.input_ref not in external_profiles
        ):
            raise ValueError(
                f"Analysis step '{step.step_id}' references missing artifact '{step.input_ref}'."
            )

    ordered: list[AnalysisStepPlan] = []
    pending = list(steps)
    available_refs = {ROOT_ARTIFACT_REF, *external_profiles}
    while pending:
        ready = [step for step in pending if step.input_ref in available_refs]
        if not ready:
            raise ValueError("Analysis step dependencies contain a cycle.")
        for step in ready:
            ordered.append(step)
            available_refs.add(step.output_ref)
            pending.remove(step)

    profiles = {ROOT_ARTIFACT_REF: root_profile, **external_profiles}
    for step in ordered:
        parent_profile = profiles[step.input_ref]
        validate_method_plan(step.method, step.n_components, step.parameters, parent_profile)
        capability = method_capability_context()[step.method]
        if step.output_role not in capability["supported_roles"]:
            raise ValueError(
                f"Method '{step.method}' does not support role '{step.output_role}'."
            )
        n_samples = int(parent_profile.get("n_samples") or 0)
        profiles[step.output_ref] = {
            "n_samples": n_samples,
            "n_features": step.n_components,
            "p_to_n_ratio": step.n_components / n_samples if n_samples else 0.0,
            "storage_type": "dense",
        }
    return ordered
