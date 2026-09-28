"""Provider-agnostic structured LLM client interfaces for DeEntropy."""

from __future__ import annotations

import json
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from agent.method_registry import (
    METHOD_CAPABILITIES,
    assess_method_feasibility,
    default_method_parameters,
    rank_feasible_methods,
)
from agent.schemas import AnalysisPlan, ReflectionDecision


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClient(ABC):
    """Minimal interface for structured high-level decisions.

    Implementations must return data that validates against the supplied
    Pydantic schema. LLM clients must not execute code or compute dataset
    statistics; DeEntropy tools own all numerical work.
    """

    @abstractmethod
    def generate_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        """Return a schema-validated structured response."""


def validate_structured_response(response: Any, schema: type[SchemaT]) -> SchemaT:
    """Validate a provider response against a Pydantic schema."""
    if isinstance(response, schema):
        return response
    if isinstance(response, str):
        response = json.loads(response)
    return schema.model_validate(response)


class StaticLLMClient(LLMClient):
    """Test helper that validates a fixed response."""

    def __init__(self, response: Any):
        self.response = response

    def generate_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        del prompt
        return validate_structured_response(self.response, schema)


class OpenAILLMClient(LLMClient):
    """OpenAI Responses API adapter with Pydantic structured outputs."""

    def __init__(
        self,
        model: str,
        temperature: float = 0,
        api_key_env: str = "OPENAI_API_KEY",
        client: Any | None = None,
    ):
        if not model or model == "configurable":
            raise ValueError("OpenAI provider requires a concrete llm.model value.")
        self.model = model
        self.temperature = temperature
        if client is not None:
            self._client = client
            return

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"OpenAI provider requires an API key in environment variable {api_key_env}."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency checked at installation
            raise RuntimeError(
                "OpenAI provider requires the 'openai' package. Install project requirements."
            ) from exc
        self._client = OpenAI(api_key=api_key)

    def generate_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        """Request structured output and validate it again at the provider boundary."""
        try:
            response = self._client.responses.parse(
                model=self.model,
                input=prompt,
                temperature=self.temperature,
                text_format=schema,
                store=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI structured request failed for {schema.__name__}: {exc}"
            ) from exc

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            parsed = getattr(response, "output_text", None)
        if parsed is None or parsed == "":
            raise ValueError(
                f"OpenAI returned no structured output for {schema.__name__}."
            )
        try:
            return validate_structured_response(parsed, schema)
        except Exception as exc:
            raise ValueError(
                f"OpenAI returned invalid structured output for {schema.__name__}: {exc}"
            ) from exc


class MockLLMClient(LLMClient):
    """Deterministic client for tests and local no-network runs."""

    def generate_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        context = _extract_context(prompt)
        if schema is AnalysisPlan:
            return validate_structured_response(_mock_plan(context), schema)
        if schema is ReflectionDecision:
            return validate_structured_response(_mock_reflection(context), schema)
        raise ValueError(f"MockLLMClient does not support schema {schema.__name__}.")


def _extract_context(prompt: str) -> dict[str, Any]:
    match = re.search(
        r"STATE_CONTEXT_JSON:\n(?P<context>.*?)\nEND_STATE_CONTEXT_JSON",
        prompt,
        flags=re.DOTALL,
    )
    if not match:
        return {}
    return json.loads(match.group("context"))


def _mock_plan(context: dict[str, Any]) -> dict[str, Any]:
    profile = context.get("dataset_profile") or {}
    sparse_input = profile.get("storage_type") == "sparse"
    remaining_budget = max(0, int(context.get("remaining_experiment_budget", 0)))
    attempted = set(context.get("attempted_methods", []))

    preprocessing = []
    if profile.get("constant_features"):
        preprocessing.append(
            {
                "action": "remove_constant_features",
                "columns": profile["constant_features"],
                "parameters": {},
                "rationale": "Remove constant features because they cannot contribute to an embedding.",
            }
        )
    numeric_features = profile.get("numeric_features") or []
    constant_features = set(profile.get("constant_features") or [])
    selectable_numeric_features = [
        column for column in numeric_features if column not in constant_features
    ]
    numeric_feature_count = len(numeric_features)
    n_samples = int(profile.get("n_samples") or 0)
    computational_concerns = profile.get("computational_concerns") or []
    if float(profile.get("missing_fraction") or 0) > 0:
        if profile.get("numeric_features"):
            preprocessing.append(
                {
                    "action": "impute_numeric",
                    "columns": None,
                    "parameters": {"strategy": "zero" if sparse_input else "median"},
                    "rationale": "Impute numeric missing values before running deterministic tools.",
                }
            )
        if profile.get("categorical_features"):
            preprocessing.append(
                {
                    "action": "impute_categorical",
                    "columns": profile["categorical_features"],
                    "parameters": {"fill_value": "__missing__"},
                    "rationale": "Preserve missing categorical values with an explicit level.",
                }
            )
    if profile.get("categorical_features"):
        preprocessing.append(
            {
                "action": "encode_categorical",
                "columns": profile["categorical_features"],
                "parameters": {"drop_first": False},
                "rationale": "Encode categorical features using the approved preprocessing tool.",
            }
        )
    count_like = bool(profile.get("count_like") and profile.get("nonnegative"))
    row_sum_range = profile.get("row_sum_range")
    totals_vary = bool(
        row_sum_range
        and float(row_sum_range[0]) >= 0
        and float(row_sum_range[1]) > max(float(row_sum_range[0]), 0.0)
    )
    if count_like and totals_vary:
        preprocessing.append(
            {
                "action": "normalize_sample_totals",
                "columns": None,
                "parameters": {"target_total": 1.0},
                "rationale": (
                    "Normalize varying nonnegative sample totals before comparing "
                    "feature magnitudes across samples."
                ),
            }
        )
    scale_or_skew_evidence = bool(
        profile.get("potential_scaling_issues")
        or float(profile.get("sparsity") or 0) >= 0.5
    )
    if count_like and scale_or_skew_evidence:
        preprocessing.append(
            {
                "action": "log_transform",
                "columns": None,
                "parameters": {"offset": 1.0},
                "rationale": (
                    "Apply sparse-safe log1p after sample-total normalization because "
                    "the count-like values show scale or sparsity-related skew evidence."
                ),
            }
        )
    if n_samples > 0 and numeric_feature_count > 0 and computational_concerns:
        top_k = min(
            len(selectable_numeric_features),
            max(n_samples, math.ceil(math.sqrt(n_samples * numeric_feature_count))),
        )
        if 0 < top_k < len(selectable_numeric_features):
            preprocessing.append(
                {
                    "action": "select_high_variance_features",
                    "columns": None,
                    "parameters": {"top_k": top_k},
                    "rationale": (
                        "Select variance-ranked features from the transformed representation "
                        "to reduce high-dimensional computation."
                    ),
                }
            )
    if not sparse_input and not count_like and (
        profile.get("numeric_features") or profile.get("categorical_features")
    ):
        preprocessing.append(
            {
                "action": "standardize",
                "columns": None,
                "parameters": {},
                "rationale": "Put usable numeric features on a comparable scale before distance-based methods.",
            }
        )

    feasibility = context.get("method_feasibility") or {
        method: assess_method_feasibility(method, profile)
        for method in METHOD_CAPABILITIES
    }
    ranked = [
        item
        for item in rank_feasible_methods(profile, feasibility)
        if item[0] not in attempted
    ]
    selected: list[tuple[str, int, str]] = []
    steps: list[dict[str, Any]] = []
    n_features = int(profile.get("n_features") or 0)
    ratio = float(profile.get("p_to_n_ratio") or 0)
    should_compose = remaining_budget >= 2 and (
        sparse_input or (ratio >= 5 and n_samples >= 500)
    )
    linear_method = "truncated_svd" if sparse_input else "pca"
    linear_is_feasible = feasibility.get(linear_method, {}).get("feasible", False)

    reusable_intermediates = [
        artifact
        for artifact in context.get("available_artifacts", [])
        if artifact.get("role") == "intermediate" and artifact.get("profile")
    ]
    if remaining_budget and reusable_intermediates:
        parent = reusable_intermediates[-1]
        parent_profile = parent["profile"]
        parent_feasibility = {
            method: assess_method_feasibility(method, parent_profile)
            for method in METHOD_CAPABILITIES
        }
        reusable_ranked = [
            item
            for item in rank_feasible_methods(parent_profile, parent_feasibility)
            if item[0] not in attempted
            and "terminal" in METHOD_CAPABILITIES[item[0]].supported_roles
        ]
        if reusable_ranked:
            method, score, reason = reusable_ranked[0]
            selected = [(method, score, reason)]
            steps = [{
                "step_id": f"terminal_{method}_{len(context.get('completed_terminal_experiments', [])) + 1}",
                "method": method,
                "input_ref": parent["artifact_id"],
                "output_ref": f"terminal_{method}_{len(context.get('completed_terminal_experiments', [])) + 1}",
                "output_role": "terminal",
                "n_components": 2,
                "parameters": default_method_parameters(method, parent_profile, n_components=2),
                "rationale": (
                    f"Reuse the completed {parent.get('method', 'intermediate')} artifact and "
                    f"apply {METHOD_CAPABILITIES[method].display_name} because {reason}."
                ),
            }]

    if not steps and should_compose and linear_is_feasible:
        rank_bound = min(n_samples, n_features)
        intermediate_components = min(
            max(3, math.ceil(math.sqrt(max(1, rank_bound)))),
            max(2, rank_bound - 1),
        )
        artifact_profile = {
            "n_samples": n_samples,
            "n_features": intermediate_components,
            "p_to_n_ratio": intermediate_components / n_samples if n_samples else 0.0,
            "storage_type": "dense",
        }
        artifact_feasibility = {
            method: assess_method_feasibility(method, artifact_profile)
            for method in METHOD_CAPABILITIES
        }
        downstream = [
            item for item in rank_feasible_methods(artifact_profile, artifact_feasibility)
            if item[0] not in attempted
            and item[0] != linear_method
            and "terminal" in METHOD_CAPABILITIES[item[0]].supported_roles
        ]
        if downstream:
            downstream_method, score, downstream_reason = downstream[0]
            selected = [
                (linear_method, 100, "a manageable dense intermediate representation is needed"),
                (downstream_method, score, downstream_reason),
            ]
            steps = [
                {
                    "step_id": "linear_intermediate",
                    "method": linear_method,
                    "input_ref": "preprocessed_X",
                    "output_ref": "linear_representation",
                    "output_role": "intermediate",
                    "n_components": intermediate_components,
                    "parameters": default_method_parameters(
                        linear_method, profile, n_components=intermediate_components
                    ),
                    "rationale": (
                        f"Use {METHOD_CAPABILITIES[linear_method].display_name} to create a "
                        "size-derived intermediate representation compatible with downstream analysis."
                    ),
                },
                {
                    "step_id": "terminal_embedding",
                    "method": downstream_method,
                    "input_ref": "linear_representation",
                    "output_ref": "terminal_embedding",
                    "output_role": "terminal",
                    "n_components": 2,
                    "parameters": default_method_parameters(
                        downstream_method, artifact_profile, n_components=2
                    ),
                    "rationale": (
                        f"Apply {METHOD_CAPABILITIES[downstream_method].display_name} to the "
                        f"dense intermediate because {downstream_reason}."
                    ),
                },
            ]

    if not steps:
        selected = ranked[: min(2, remaining_budget)]
        steps = [
            {
                "step_id": f"{method}_{index}",
                "method": method,
                "input_ref": "preprocessed_X",
                "output_ref": f"artifact_{method}_{index}",
                "output_role": "terminal",
                "n_components": 2,
                "parameters": default_method_parameters(method, profile, n_components=2),
                "rationale": f"Select {METHOD_CAPABILITIES[method].display_name} because {reason}.",
            }
            for index, (method, _, reason) in enumerate(selected, start=1)
        ]
    methods = [
        {
            "method": step["method"],
            "n_components": step["n_components"],
            "parameters": step["parameters"],
            "rationale": step["rationale"],
        }
        for step in steps
    ]
    selected_names = {item[0] for item in selected}
    alternatives = [
        (
            f"Did not select {METHOD_CAPABILITIES[method].display_name}: it was feasible, "
            "but higher-priority profile-matched methods filled the remaining experiment budget."
        )
        for method, _, _ in ranked
        if method not in selected_names
    ][:3]

    return {
        "preprocessing": preprocessing,
        "methods": methods,
        "steps": steps,
        "evaluation_metrics": [
            "trustworthiness",
            "knn_preservation",
            "distance_pearson_correlation",
            "distance_spearman_correlation",
        ],
        "analysis_goal": "Produce a small, justified set of embeddings for exploratory dimension reduction.",
        "rationale": (
            "The plan ranks executable, feasible methods from observed sample size, feature "
            "dimensionality, computational scaling, previous attempts, and remaining budget."
        ),
        "alternatives_not_selected": alternatives,
    }

def _mock_reflection(context: dict[str, Any]) -> dict[str, Any]:
    if context.get("iteration_limit_reached"):
        return {
            "decision": "stop",
            "reason": "Maximum iteration budget has been reached.",
            "suggested_changes": [],
            "preferred_embedding_id": context.get("best_embedding_id"),
        }
    if context.get("experiment_budget_exhausted"):
        return {
            "decision": "stop",
            "reason": "Experiment budget is exhausted.",
            "suggested_changes": [],
            "preferred_embedding_id": context.get("best_embedding_id"),
        }
    if context.get("analysis_step_limit_exhausted"):
        return {
            "decision": "stop",
            "reason": "Analysis-step safety limit is exhausted.",
            "suggested_changes": [],
            "preferred_embedding_id": context.get("best_embedding_id"),
        }

    best = context.get("best_evaluation") or {}
    metrics = best.get("metrics") or {}
    trust = metrics.get("trustworthiness")
    knn = metrics.get("knn_preservation")
    best_id = best.get("embedding_id")

    if trust is not None and knn is not None and trust >= 0.8 and knn >= 0.5:
        return {
            "decision": "stop",
            "reason": "Current best embedding has acceptable local-structure metrics.",
            "suggested_changes": [],
            "preferred_embedding_id": best_id,
        }

    return {
        "decision": "continue",
        "reason": "Current results do not yet show clearly acceptable local-structure preservation.",
        "suggested_changes": ["try method umap", "adjust n_neighbors"],
        "preferred_embedding_id": best_id,
    }
