import pytest
from pydantic import ValidationError

from agent.llm_client import MockLLMClient, StaticLLMClient
from agent.reflector import build_reflector_context, reflect, synthesize_completed_results
from agent.schemas import DatasetProfile
from agent.state import AgentState


def sample_profile():
    return DatasetProfile(
        n_samples=30,
        n_features=5,
        numeric_features=["x1", "x2"],
        categorical_features=[],
        boolean_features=[],
        missing_fraction=0.0,
        missingness_by_column={},
        sparsity=0.0,
        p_to_n_ratio=1 / 6,
        constant_features=[],
        near_constant_features=[],
        duplicate_rows=0,
        numeric_summaries={},
        potential_scaling_issues=[],
        high_cardinality_categorical_features=[],
        has_labels=False,
        label_column=None,
        warnings=[],
    )


def state_with_evaluation(metrics):
    state = AgentState(dataset_path="data.csv", dataset_profile=sample_profile())
    state.record_analysis({"method": "pca", "embedding_id": "pca_1"})
    state.record_evaluation({"embedding_id": "pca_1", "metrics": metrics})
    return state


def test_reflector_stops_for_good_metrics():
    state = state_with_evaluation({"trustworthiness": 0.9, "knn_preservation": 0.7})

    decision = reflect(state, MockLLMClient())

    assert decision.decision == "stop"
    assert decision.preferred_embedding_id == "pca_1"
    assert state.finished is True


def test_reflector_continues_for_weak_metrics_with_valid_suggestions():
    state = state_with_evaluation({"trustworthiness": 0.55, "knn_preservation": 0.2})

    decision = reflect(state, MockLLMClient())

    assert decision.decision == "continue"
    assert state.finished is False
    assert decision.suggested_changes
    assert any("umap" in suggestion for suggestion in decision.suggested_changes)


def test_reflector_respects_iteration_limit_without_llm_decision():
    state = state_with_evaluation({"trustworthiness": 0.1, "knn_preservation": 0.1})
    state.iteration = state.max_iterations

    decision = reflect(state, MockLLMClient())

    assert decision.decision == "stop"
    assert "iteration" in decision.reason.lower()
    assert state.finished is True


def test_reflector_respects_experiment_budget_without_llm_decision():
    state = AgentState(
        dataset_path="data.csv",
        dataset_profile=sample_profile(),
        max_experiments=1,
    )
    state.record_analysis({"method": "pca", "embedding_id": "pca_1"})

    decision = reflect(state, MockLLMClient())

    assert decision.decision == "stop"
    assert "budget" in decision.reason.lower()


def test_reflector_counts_terminal_experiments_not_intermediate_artifacts():
    state = AgentState(
        dataset_path="data.csv",
        dataset_profile=sample_profile(),
        max_experiments=1,
    )
    state.record_analysis({
        "method": "pca", "embedding_id": "pca_k", "output_role": "intermediate",
        "execution_status": "completed",
    })

    context = build_reflector_context(state)

    assert context["experiment_count"] == 0
    assert context["experiment_budget_exhausted"] is False
    assert context["terminal_experiment_history"] == []


def test_budget_exhaustion_still_records_metric_aware_recommendation():
    state = AgentState(
        dataset_path="data.csv",
        dataset_profile=sample_profile(),
        max_experiments=2,
    )
    state.record_analysis({"method": "first", "embedding_id": "local_first"})
    state.record_analysis({"method": "second", "embedding_id": "latest_global"})
    state.record_evaluation(
        {
            "embedding_id": "local_first",
            "intrinsic_metrics": {
                "trustworthiness": 0.95,
                "knn_preservation": 0.8,
                "distance_pearson_correlation": 0.4,
                "distance_spearman_correlation": 0.45,
            },
        }
    )
    state.record_evaluation(
        {
            "embedding_id": "latest_global",
            "intrinsic_metrics": {
                "trustworthiness": 0.85,
                "knn_preservation": 0.5,
                "distance_pearson_correlation": 0.9,
                "distance_spearman_correlation": 0.92,
            },
        }
    )

    decision = reflect(state, MockLLMClient())

    assert decision.decision == "stop"
    assert decision.preferred_embedding_id == "local_first"
    assert state.best_embedding_id == "local_first"
    assert state.final_recommendation is not None
    purposes = {
        item.purpose: item.embedding_id
        for item in state.final_recommendation.purpose_recommendations
    }
    assert purposes["local-neighborhood exploration"] == "local_first"
    assert purposes["broader pairwise-structure reference"] == "latest_global"
    assert state.final_recommendation.tradeoffs


def test_synthesis_does_not_average_heterogeneous_metrics():
    state = AgentState(dataset_path="data.csv", dataset_profile=sample_profile())
    state.record_evaluation(
        {
            "embedding_id": "local",
            "intrinsic_metrics": {
                "knn_preservation": 0.9,
                "trustworthiness": 0.95,
                "distance_spearman_correlation": 0.2,
                "distance_pearson_correlation": 0.2,
            },
        }
    )
    state.record_evaluation(
        {
            "embedding_id": "global",
            "intrinsic_metrics": {
                "knn_preservation": 0.4,
                "trustworthiness": 0.8,
                "distance_spearman_correlation": 0.9,
                "distance_pearson_correlation": 0.9,
            },
        }
    )

    synthesis = synthesize_completed_results(state)

    assert synthesis.primary_embedding_id == "local"
    assert len(synthesis.purpose_recommendations) == 2
    assert synthesis.tradeoffs


def test_reflector_context_selects_best_actual_evaluation():
    state = AgentState(dataset_path="data.csv", dataset_profile=sample_profile())
    state.record_evaluation(
        {"embedding_id": "weak", "metrics": {"trustworthiness": 0.2, "knn_preservation": 0.2}}
    )
    state.record_evaluation(
        {"embedding_id": "strong", "metrics": {"trustworthiness": 0.9, "knn_preservation": 0.8}}
    )

    context = build_reflector_context(state)

    assert context["best_evaluation"]["embedding_id"] == "strong"


def test_invalid_llm_reflection_response_is_rejected():
    client = StaticLLMClient(
        {
            "decision": "maybe",
            "reason": "Invalid.",
            "suggested_changes": [],
            "preferred_embedding_id": None,
        }
    )
    state = state_with_evaluation({"trustworthiness": 0.2, "knn_preservation": 0.2})

    with pytest.raises(ValidationError):
        reflect(state, client)


def test_reflector_rejects_invalid_suggested_next_actions():
    client = StaticLLMClient(
        {
            "decision": "continue",
            "reason": "Try something else.",
            "suggested_changes": ["execute arbitrary python"],
            "preferred_embedding_id": None,
        }
    )
    state = state_with_evaluation({"trustworthiness": 0.2, "knn_preservation": 0.2})

    with pytest.raises(ValueError, match="invalid next actions"):
        reflect(state, client)
