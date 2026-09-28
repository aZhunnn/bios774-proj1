import pytest
from pydantic import ValidationError

from agent.llm_client import MockLLMClient, StaticLLMClient
from agent.planner import build_planner_context, create_plan
from agent.schemas import DatasetProfile
from agent.state import AgentState


def sample_profile(**updates):
    data = {
        "n_samples": 30,
        "n_features": 5,
        "numeric_features": ["x1", "x2", "x3"],
        "categorical_features": [],
        "boolean_features": [],
        "missing_fraction": 0.0,
        "missingness_by_column": {},
        "sparsity": 0.0,
        "p_to_n_ratio": 1 / 6,
        "constant_features": [],
        "near_constant_features": [],
        "duplicate_rows": 0,
        "numeric_summaries": {},
        "potential_scaling_issues": [],
        "high_cardinality_categorical_features": [],
        "has_labels": False,
        "label_column": None,
        "warnings": [],
    }
    data.update(updates)
    return DatasetProfile(**data)


def test_planner_consumes_compact_state_and_returns_valid_plan():
    state = AgentState(dataset_path="data.csv", dataset_profile=sample_profile())

    plan = create_plan(state, MockLLMClient())

    assert state.current_plan == plan
    assert 1 <= len(plan.methods) <= 2
    assert {method.method for method in plan.methods} == {"mds", "kernel_pca"}
    assert all(method.parameters for method in plan.methods)
    assert "trustworthiness" in plan.evaluation_metrics
    assert plan.alternatives_not_selected


def test_planner_adds_preprocessing_for_profile_needs():
    state = AgentState(
        dataset_path="data.csv",
        dataset_profile=sample_profile(
            categorical_features=["group"],
            constant_features=["constant"],
            missing_fraction=0.2,
        ),
    )

    plan = create_plan(state, MockLLMClient())
    actions = [action.action for action in plan.preprocessing]

    assert "remove_constant_features" in actions
    assert "impute_numeric" in actions
    assert "impute_categorical" in actions
    assert "encode_categorical" in actions
    assert "standardize" in actions


def test_planner_respects_remaining_experiment_budget():
    state = AgentState(
        dataset_path="data.csv",
        dataset_profile=sample_profile(),
        max_experiments=1,
    )

    plan = create_plan(state, MockLLMClient())

    assert len(plan.methods) == 1


def test_planner_can_choose_profile_driven_top_k_feature_selection():
    numeric_features = [f"x{index}" for index in range(600)]
    state = AgentState(
        dataset_path="data.csv",
        dataset_profile=sample_profile(
            n_samples=50,
            n_features=600,
            numeric_features=numeric_features,
            p_to_n_ratio=12,
            computational_concerns=["High-dimensional computation."],
        ),
    )

    plan = create_plan(state, MockLLMClient())
    selection = next(
        action
        for action in plan.preprocessing
        if action.action == "select_high_variance_features"
    )

    assert 0 < selection.parameters["top_k"] < 600
    assert selection.parameters["top_k"] >= 50


def test_high_dimensional_profile_selects_different_method_subset():
    numeric_features = [f"x{index}" for index in range(600)]
    state = AgentState(
        dataset_path="data.csv",
        dataset_profile=sample_profile(
            n_samples=50,
            n_features=600,
            numeric_features=numeric_features,
            p_to_n_ratio=12,
            computational_concerns=["High-dimensional computation."],
        ),
    )

    plan = create_plan(state, MockLLMClient())

    assert [method.method for method in plan.methods] == ["sparse_pca", "pca"]
    assert all("feature" in method.rationale.lower() for method in plan.methods)


def test_larger_n_profile_excludes_quadratic_methods():
    numeric_features = [f"x{index}" for index in range(20)]
    state = AgentState(
        dataset_path="data.csv",
        dataset_profile=sample_profile(
            n_samples=3000,
            n_features=20,
            numeric_features=numeric_features,
            p_to_n_ratio=20 / 3000,
        ),
    )

    context = build_planner_context(state)
    plan = create_plan(state, MockLLMClient())

    assert [method.method for method in plan.methods] == ["umap", "pca"]
    assert context["method_feasibility"]["mds"]["feasible"] is False
    assert context["method_feasibility"]["kernel_pca"]["feasible"] is False


def test_sparse_profile_proposes_sparse_linear_intermediate_and_dense_terminal():
    state = AgentState(
        dataset_path="matrix.mtx",
        dataset_profile=sample_profile(
            n_samples=100,
            n_features=1000,
            numeric_features=[f"f{index}" for index in range(1000)],
            p_to_n_ratio=10,
            constant_features=["f0"],
            computational_concerns=["Preserve sparse storage."],
            storage_type="sparse",
            sparse_format="csr",
        ),
    )

    plan = create_plan(state, MockLLMClient())
    actions = {action.action for action in plan.preprocessing}

    assert [step.method for step in plan.steps] == ["truncated_svd", "mds"]
    assert plan.steps[0].output_role == "intermediate"
    assert plan.steps[1].input_ref == plan.steps[0].output_ref
    assert plan.steps[1].output_role == "terminal"
    assert "remove_constant_features" in actions
    assert "select_high_variance_features" in actions
    assert "standardize" not in actions


def test_count_like_profile_receives_ordered_count_transform_consideration():
    state = AgentState(
        dataset_path="counts.mtx",
        dataset_profile=sample_profile(
            n_samples=100,
            n_features=1000,
            numeric_features=[f"f{i}" for i in range(1000)],
            p_to_n_ratio=10,
            storage_type="sparse",
            sparse_format="csr",
            nonnegative=True,
            integer_valued=True,
            count_like=True,
            row_sum_range=(10.0, 1000.0),
            sparsity=0.9,
            computational_concerns=["High-dimensional computation."],
        ),
    )

    plan = create_plan(state, MockLLMClient())
    actions = [action.action for action in plan.preprocessing]

    assert "normalize_sample_totals" in actions
    assert "log_transform" in actions
    assert actions.index("normalize_sample_totals") < actions.index("log_transform")
    assert actions.index("log_transform") < actions.index("select_high_variance_features")


def test_non_count_continuous_profile_does_not_receive_count_transforms():
    state = AgentState(
        dataset_path="continuous.csv",
        dataset_profile=sample_profile(
            nonnegative=True,
            integer_valued=False,
            count_like=False,
            row_sum_range=(1.0, 100.0),
            potential_scaling_issues=["x1", "x2"],
        ),
    )

    plan = create_plan(state, MockLLMClient())
    actions = {action.action for action in plan.preprocessing}

    assert "normalize_sample_totals" not in actions
    assert "log_transform" not in actions


def test_planner_rejects_variance_selection_before_count_transform():
    client = StaticLLMClient(
        {
            "preprocessing": [
                {"action": "select_high_variance_features", "parameters": {"top_k": 2}, "rationale": "Too early."},
                {"action": "normalize_sample_totals", "parameters": {"target_total": 1.0}, "rationale": "Too late."},
            ],
            "methods": [],
            "evaluation_metrics": [],
            "analysis_goal": "Validate order.",
            "rationale": "Validate order.",
        }
    )
    state = AgentState(dataset_path="counts.csv", dataset_profile=sample_profile())

    with pytest.raises(ValueError, match="must follow"):
        create_plan(state, client)


def test_previously_attempted_methods_are_not_repeated_when_alternatives_exist():
    state = AgentState(dataset_path="data.csv", dataset_profile=sample_profile())
    state.record_analysis({"method": "mds", "embedding_id": "mds_1"})

    plan = create_plan(state, MockLLMClient())

    assert "mds" not in {method.method for method in plan.methods}
    assert plan.methods


def test_planner_rejects_invalid_neighborhood_size():
    client = StaticLLMClient(
        {
            "preprocessing": [],
            "methods": [{
                "method": "isomap",
                "n_components": 2,
                "parameters": {"n_neighbors": 30},
                "rationale": "Invalid neighborhood.",
            }],
            "evaluation_metrics": ["trustworthiness"],
            "analysis_goal": "Validate.",
            "rationale": "Validate.",
        }
    )
    state = AgentState(dataset_path="data.csv", dataset_profile=sample_profile())

    with pytest.raises(ValueError, match="n_neighbors"):
        create_plan(state, client)


def test_planner_rejects_invalid_tsne_perplexity():
    client = StaticLLMClient(
        {
            "preprocessing": [],
            "methods": [{
                "method": "tsne",
                "n_components": 2,
                "parameters": {"perplexity": 30},
                "rationale": "Invalid perplexity.",
            }],
            "evaluation_metrics": ["trustworthiness"],
            "analysis_goal": "Validate.",
            "rationale": "Validate.",
        }
    )
    state = AgentState(dataset_path="data.csv", dataset_profile=sample_profile())

    with pytest.raises(ValueError, match="perplexity"):
        create_plan(state, client)


def test_planner_returns_no_methods_when_budget_exhausted():
    state = AgentState(
        dataset_path="data.csv",
        dataset_profile=sample_profile(),
        max_experiments=1,
    )
    state.record_analysis({"method": "pca", "embedding_id": "pca_1"})

    plan = create_plan(state, MockLLMClient())

    assert plan.methods == []


def test_planner_context_does_not_invent_dataset_statistics():
    state = AgentState(dataset_path="data.csv", dataset_profile=sample_profile(n_samples=17))

    context = build_planner_context(state)

    assert context["dataset_profile"]["n_samples"] == 17
    assert context["remaining_experiment_budget"] == 10


def test_invalid_llm_plan_response_is_rejected():
    client = StaticLLMClient(
        {
            "preprocessing": [{"action": "unknown", "rationale": "Bad."}],
            "methods": [],
            "evaluation_metrics": [],
            "analysis_goal": "Invalid.",
            "rationale": "Invalid.",
        }
    )
    state = AgentState(dataset_path="data.csv", dataset_profile=sample_profile())

    with pytest.raises(ValidationError):
        create_plan(state, client)


def test_planner_rejects_unavailable_but_schema_allowed_dr_method():
    client = StaticLLMClient(
        {
            "preprocessing": [],
            "methods": [
                {
                    "method": "gplvm",
                    "n_components": 2,
                    "parameters": {},
                    "rationale": "Not available in Phase 3.",
                }
            ],
            "evaluation_metrics": ["trustworthiness"],
            "analysis_goal": "Invalid.",
            "rationale": "Uses unavailable method.",
        }
    )
    state = AgentState(dataset_path="data.csv", dataset_profile=sample_profile())

    with pytest.raises(ValueError, match="unavailable DR method"):
        create_plan(state, client)
