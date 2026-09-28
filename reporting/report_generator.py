"""Deterministic Markdown report generation for DeEntropy V1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.reflector import synthesize_completed_results
from agent.state import AgentState


def generate_report(
    state: AgentState,
    output_dir: str | Path,
    dataset_name: str,
    config: dict[str, Any] | None = None,
) -> Path:
    """Write a Markdown report using only stored state and result histories."""
    config = config or {}
    output_path = Path(output_dir) / report_filename(dataset_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(_report_lines(state, dataset_name, config)) + "\n",
        encoding="utf-8",
    )
    return output_path


def report_filename(dataset_name: str) -> str:
    suffix = dataset_name.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return f"generated_report_{suffix}.md"
    return "generated_report.md"


def _report_lines(
    state: AgentState,
    dataset_name: str,
    config: dict[str, Any],
) -> list[str]:
    profile = state.dataset_profile
    lines = [
        f"# DeEntropy Analysis Report: {dataset_name}",
        "",
        "## Dataset Inspection",
    ]
    if profile is None:
        lines.extend(["No dataset profile was recorded.", ""])
    else:
        lines.extend(
            [
                f"- Samples: {profile.n_samples}",
                f"- Features: {profile.n_features}",
                f"- Numeric features: {len(profile.numeric_features)}",
                f"- Categorical features: {len(profile.categorical_features)}",
                f"- Boolean features: {len(profile.boolean_features)}",
                f"- Missing-value fraction: {_fmt(profile.missing_fraction)}",
                f"- Sparsity: {_fmt(profile.sparsity)}",
                f"- p/n ratio: {_fmt(profile.p_to_n_ratio)}",
                f"- Storage: {profile.storage_type}",
                f"- Nonzero entries: {_fmt(profile.nonzero_count)}",
                f"- Density: {_fmt(profile.density)}",
                f"- Label column: {profile.label_column if profile.has_labels else 'None'}",
                "",
            ]
        )
        if profile.constant_features or profile.near_constant_features or profile.warnings:
            lines.append("Important data characteristics:")
            lines.extend(_bullet_list(profile.warnings or ["No profile warnings recorded."]))
            lines.append("")

    lines.extend(["## Preprocessing", ""])
    if state.preprocessing_history:
        seen_actions: set[tuple[str, str]] = set()
        strategies: list[str] = []
        for item in state.preprocessing_history:
            strategy = str(item.get("strategy_id", "legacy"))
            key = (strategy, str(item.get("action")))
            if key in seen_actions:
                continue
            seen_actions.add(key)
            if strategy not in strategies:
                strategies.append(strategy)
                lines.append(
                    f"- Pipeline `{strategy}`: `{item.get('input_ref', 'raw_X')}` -> "
                    f"`{item.get('output_ref', 'preprocessed_X')}`"
                )
            lines.append(
                f"  - {item.get('action')}: {_column_summary(item.get('columns'))}; "
                f"status={item.get('status', 'unknown')}"
            )
            if item.get("parameters"):
                lines.append(f"    Parameters: `{_compact_value(item['parameters'])}`")
            if item.get("rationale"):
                lines.append(f"    Rationale: {item['rationale']}")
    else:
        lines.append("No preprocessing actions were recorded.")
    lines.append("")

    lines.extend(["## Methods and Rationales", ""])
    if state.analysis_history:
        for item in state.analysis_history:
            lines.append(
                f"- Iteration {item.get('iteration', 'n/a')}: {item.get('method')} "
                f"({item.get('embedding_id')}); success={item.get('success')}"
            )
            lines.append(f"  Rationale: {item.get('rationale', 'Not recorded.')}")
            lines.append(f"  Parameters: `{_compact_value(item.get('parameters', {}))}`")
            if item.get("error_message"):
                lines.append(f"  Error: {item['error_message']}")
    else:
        lines.append("No dimension-reduction methods were recorded.")
    lines.append("")

    lines.extend(["## Intrinsic Embedding Evaluation", ""])
    if state.evaluation_history:
        for item in state.evaluation_history:
            lines.append(f"- Embedding: {item.get('embedding_id')}")
            intrinsic = item.get("intrinsic_metrics") or item.get("metrics") or {}
            for metric, value in intrinsic.items():
                lines.append(f"  - {metric}: {_fmt(value)}")
            for warning in item.get("warnings", []):
                lines.append(f"  - Warning: {warning}")
            for note in item.get("notes", []):
                lines.append(f"  - Note: {note}")
    else:
        lines.append("No successful embedding evaluations were recorded.")
    lines.append("")

    lines.extend(["## External Label-Based Validation", ""])
    external_results = [
        item for item in state.evaluation_history if item.get("external_validation_metrics")
    ]
    if external_results:
        lines.append(
            "Labels were used only for post-hoc validation and visualization, not as "
            "dimension-reduction features, preprocessing targets, or tuning objectives."
        )
        lines.append("")
        for item in external_results:
            lines.append(f"- Embedding: {item.get('embedding_id')}")
            for metric, value in item["external_validation_metrics"].items():
                lines.append(f"  - {metric}: {_fmt(value)}")
    else:
        lines.append("No label-based external validation metrics were available.")
    lines.append("")

    lines.extend(["## Agent Reflection", ""])
    if state.reflection_history:
        for index, item in enumerate(state.reflection_history, start=1):
            lines.append(
                f"- Reflection {index}: {item.get('decision')} - {item.get('reason')}"
            )
            if item.get("preferred_embedding_id"):
                lines.append(f"  Preferred embedding: {item['preferred_embedding_id']}")
            if item.get("suggested_changes"):
                lines.append(f"  Suggested changes: {', '.join(item['suggested_changes'])}")
    else:
        lines.append("No reflection decisions were recorded.")
    lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            "- Automated method selection is limited to the configured V1 tool set.",
            "- Metrics summarize structural preservation but do not prove biological or domain significance.",
            "- Label alignment can describe correspondence with known classes but cannot establish mechanisms or causal explanations.",
            "- Stochastic methods may vary despite controlled random seeds.",
        ]
    )
    if state.warnings:
        lines.extend(_bullet_list(state.warnings))
    if state.errors:
        lines.extend(_bullet_list(state.errors))
    lines.append("")

    lines.extend(["## Final Recommendation", ""])
    synthesis = state.final_recommendation or synthesize_completed_results(state)
    if synthesis:
        lines.append(synthesis.summary)
        lines.append("")
        for recommendation in synthesis.purpose_recommendations:
            lines.append(
                f"- For {recommendation.purpose}: `{recommendation.embedding_id}`. "
                f"{recommendation.basis}"
            )
        if synthesis.tradeoffs:
            lines.append("")
            lines.append("Tradeoffs:")
            lines.extend(_bullet_list(synthesis.tradeoffs))
    else:
        lines.append("No successful embedding was available for recommendation.")
    lines.append("")

    lines.extend(
        [
            "## Reproducibility",
            "",
            f"- Random seed: {config.get('random_seed', 'not recorded')}",
            f"- Max iterations: {state.max_iterations}",
            f"- Max experiments: {state.max_experiments}",
        ]
    )
    return lines


def _bullet_list(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def _fmt(value: Any) -> str:
    if value is None:
        return "not computed"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _column_summary(columns: Any) -> str:
    if not columns:
        return "all applicable features"
    values = list(columns)
    preview = ", ".join(map(str, values[:3]))
    suffix = "" if len(values) <= 3 else ", ..."
    return f"{len(values)} feature(s) [{preview}{suffix}]"


def _compact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _compact_value(item) for key, item in value.items()}
    if isinstance(value, list) and len(value) > 8:
        return f"<{len(value)} values>"
    if isinstance(value, list):
        return [_compact_value(item) for item in value]
    return value
