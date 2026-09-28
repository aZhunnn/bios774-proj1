import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse
from scipy.io import mmwrite

from agent.llm_client import LLMClient, validate_structured_response
from agent.planner import create_plan, validate_and_order_steps
from agent.schemas import AnalysisPlan, AnalysisStepPlan, EmbeddingResult, ReflectionDecision
from agent.state import AgentState
from main import run_agent


def root_profile(storage_type="dense", n_samples=30, n_features=8):
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "p_to_n_ratio": n_features / n_samples,
        "storage_type": storage_type,
    }


def step(step_id, method, input_ref, output_ref, role="terminal", n_components=2):
    return AnalysisStepPlan(
        step_id=step_id,
        method=method,
        input_ref=input_ref,
        output_ref=output_ref,
        output_role=role,
        n_components=n_components,
        parameters={},
        rationale="test graph",
    )


def test_independent_branches_and_chained_steps_are_topologically_ordered():
    steps = [
        step("child", "mds", "linear", "visual"),
        step("independent", "pca", "preprocessed_X", "pca_2d"),
        step("parent", "pca", "preprocessed_X", "linear", "intermediate", 4),
    ]

    ordered = validate_and_order_steps(steps, root_profile())

    assert [item.step_id for item in ordered] == ["independent", "parent", "child"]
    assert ordered[-1].input_ref == "linear"


def test_missing_artifact_and_cycles_are_rejected():
    with pytest.raises(ValueError, match="missing artifact"):
        validate_and_order_steps(
            [step("bad", "pca", "absent", "result")], root_profile()
        )

    cyclic = [
        step("a", "pca", "artifact_b", "artifact_a"),
        step("b", "pca", "artifact_a", "artifact_b"),
    ]
    with pytest.raises(ValueError, match="cycle"):
        validate_and_order_steps(cyclic, root_profile())


def test_method_role_is_validated():
    with pytest.raises(ValueError, match="does not support role"):
        validate_and_order_steps(
            [step("mds", "mds", "preprocessed_X", "mds_k", "intermediate")],
            root_profile(),
        )


class GraphClient(LLMClient):
    def __init__(self, plan):
        self.plan = plan

    def generate_structured(self, prompt, schema):
        del prompt
        if schema is AnalysisPlan:
            return validate_structured_response(self.plan, schema)
        if schema is ReflectionDecision:
            return ReflectionDecision(decision="stop", reason="Integration test complete.")
        raise AssertionError(schema)


def config(path, data, max_experiments=2, max_analysis_steps=20):
    payload = {
        "random_seed": 42,
        "agent": {
            "max_iterations": 1,
            "max_experiments": max_experiments,
            "max_analysis_steps": max_analysis_steps,
        },
        "llm": {"provider": "mock"},
        "data": data,
        "output": {"root": str(path.parent / "outputs")},
        "evaluation": {"trustworthiness_neighbors": 3, "distance_sample_size": 30},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def graph_plan(linear_method):
    return {
        "preprocessing": [],
        "methods": [],
        "steps": [
            {
                "step_id": "linear",
                "method": linear_method,
                "input_ref": "preprocessed_X",
                "output_ref": "linear_k",
                "output_role": "intermediate",
                "n_components": 4,
                "parameters": {},
                "rationale": "Create a reusable linear representation.",
            },
            {
                "step_id": "visual",
                "method": "mds",
                "input_ref": "linear_k",
                "output_ref": "visual_2d",
                "output_role": "terminal",
                "n_components": 2,
                "parameters": {"n_init": 1, "max_iter": 20},
                "rationale": "Create a terminal view from the intermediate.",
            },
        ],
        "evaluation_metrics": ["trustworthiness"],
        "analysis_goal": "Test a composed workflow.",
        "rationale": "Exercise artifact dependencies.",
    }


def test_dense_pca_intermediate_executes_before_terminal_and_records_provenance(tmp_path):
    data_path = tmp_path / "data.csv"
    config_path = tmp_path / "config.yaml"
    rng = np.random.default_rng(42)
    pd.DataFrame(rng.normal(size=(24, 8))).to_csv(data_path, index=False)
    config(config_path, {"format": "csv"})

    state = run_agent(data_path, "dense_graph", config_path, GraphClient(graph_plan("pca")))

    assert [item["step_id"] for item in state.analysis_history] == ["linear", "visual"]
    assert state.analysis_history[0]["output_role"] == "intermediate"
    assert state.analysis_steps_used() == 2
    assert state.experiments_used() == 1
    assert len(state.evaluation_history) == 1
    evaluation = state.evaluation_history[0]
    assert evaluation["provenance"]["parent_fidelity_reference"] == "linear_k"
    assert evaluation["root_fidelity"]["status"] == "computed"


def test_sparse_svd_chain_preserves_sparse_root_and_distinguishes_fidelity(tmp_path):
    matrix_path = tmp_path / "matrix.mtx"
    samples_path = tmp_path / "samples.tsv"
    features_path = tmp_path / "features.tsv"
    config_path = tmp_path / "config.yaml"
    rng = np.random.default_rng(7)
    matrix = sparse.random(24, 12, density=0.2, random_state=rng, format="csr")
    mmwrite(matrix_path, matrix)
    samples_path.write_text("\n".join(f"s{i}" for i in range(24)) + "\n", encoding="utf-8")
    features_path.write_text("\n".join(f"f{i}" for i in range(12)) + "\n", encoding="utf-8")
    config(
        config_path,
        {
            "format": "matrix_market",
            "orientation": "samples_by_features",
            "sample_metadata_path": str(samples_path),
            "feature_metadata_path": str(features_path),
            "sample_id_column": 0,
            "feature_id_column": 0,
        },
    )

    state = run_agent(
        matrix_path, "sparse_graph", config_path, GraphClient(graph_plan("truncated_svd"))
    )

    assert state.analysis_history[0]["metadata"]["input_was_sparse"] is True
    assert state.analysis_history[1]["success"] is True
    assert state.evaluation_history[0]["root_fidelity"]["status"] == "not_computed"
    assert state.evaluation_history[0]["parent_fidelity"]["metrics"]["trustworthiness"] is not None


def test_failed_parent_blocks_child_without_consuming_second_budget_slot(tmp_path, monkeypatch):
    data_path = tmp_path / "data.csv"
    config_path = tmp_path / "config.yaml"
    pd.DataFrame(np.arange(120).reshape(20, 6)).to_csv(data_path, index=False)
    config(config_path, {"format": "csv"})

    def fail_parent(X, method, n_components, parameters, random_seed):
        del X, n_components, parameters, random_seed
        return EmbeddingResult(
            embedding_id="failed", method=method, n_components=4, runtime_seconds=0,
            success=False, error_message="forced parent failure",
        )

    monkeypatch.setattr("main.run_dimension_reduction", fail_parent)
    state = run_agent(data_path, "failed_graph", config_path, GraphClient(graph_plan("pca")))

    assert state.analysis_history[0]["execution_status"] == "failed"
    assert state.analysis_history[1]["execution_status"] == "blocked_dependency"
    assert state.experiments_used() == 0
    assert state.analysis_steps_used() == 1
    assert state.evaluation_history == []


def test_one_chained_terminal_workflow_counts_as_one_experiment():
    state = AgentState(dataset_path="data.csv", max_experiments=1)
    from agent.schemas import DatasetProfile

    state.dataset_profile = DatasetProfile(
        n_samples=30, n_features=8, numeric_features=[f"f{i}" for i in range(8)],
        categorical_features=[], boolean_features=[], missing_fraction=0, sparsity=0,
        p_to_n_ratio=8 / 30, constant_features=[], near_constant_features=[],
        has_labels=False, label_column=None, warnings=[],
    )
    plan = create_plan(state, GraphClient(graph_plan("pca")))

    assert [step.step_id for step in plan.steps] == ["linear", "visual"]


def test_shared_intermediate_supports_two_terminal_experiments_with_three_steps():
    plan = graph_plan("pca")
    plan["steps"].append({
        "step_id": "visual_two",
        "method": "tsne",
        "input_ref": "linear_k",
        "output_ref": "visual_two_2d",
        "output_role": "terminal",
        "n_components": 2,
        "parameters": {"perplexity": 5.0, "max_iter": 250},
        "rationale": "Second terminal strategy sharing the same parent.",
    })
    state = AgentState(dataset_path="data.csv", max_experiments=2)
    from agent.schemas import DatasetProfile
    state.dataset_profile = DatasetProfile(
        n_samples=30, n_features=8, numeric_features=[f"f{i}" for i in range(8)],
        categorical_features=[], boolean_features=[], missing_fraction=0, sparsity=0,
        p_to_n_ratio=8 / 30, constant_features=[], near_constant_features=[],
        has_labels=False, label_column=None, warnings=[],
    )

    selected = create_plan(state, GraphClient(plan))

    assert [step.step_id for step in selected.steps] == ["linear", "visual", "visual_two"]
    assert sum(step.output_role == "terminal" for step in selected.steps) == 2


def test_shared_intermediate_executes_once_for_two_terminal_children(tmp_path):
    data_path = tmp_path / "data.csv"
    config_path = tmp_path / "config.yaml"
    rng = np.random.default_rng(11)
    pd.DataFrame(rng.normal(size=(24, 8))).to_csv(data_path, index=False)
    config(config_path, {"format": "csv"}, max_experiments=2)
    plan = graph_plan("pca")
    plan["steps"].append({
        "step_id": "visual_two", "method": "mds", "input_ref": "linear_k",
        "output_ref": "visual_two_2d", "output_role": "terminal", "n_components": 2,
        "parameters": {"n_init": 1, "max_iter": 25}, "rationale": "Second terminal.",
    })

    state = run_agent(data_path, "shared_graph", config_path, GraphClient(plan))

    assert [item["step_id"] for item in state.analysis_history].count("linear") == 1
    assert state.analysis_steps_used() == 3
    assert state.experiments_used() == 2
    assert len(state.evaluation_history) == 2


def test_terminal_budget_retains_shared_parent_and_only_allowed_terminal():
    plan = graph_plan("pca")
    plan["steps"].append({
        "step_id": "visual_two", "method": "tsne", "input_ref": "linear_k",
        "output_ref": "visual_two_2d", "output_role": "terminal", "n_components": 2,
        "parameters": {"perplexity": 5.0, "max_iter": 250}, "rationale": "Second terminal.",
    })
    state = AgentState(dataset_path="data.csv", max_experiments=1)
    from agent.schemas import DatasetProfile
    state.dataset_profile = DatasetProfile(
        n_samples=30, n_features=8, numeric_features=[f"f{i}" for i in range(8)],
        categorical_features=[], boolean_features=[], missing_fraction=0, sparsity=0,
        p_to_n_ratio=8 / 30, constant_features=[], near_constant_features=[],
        has_labels=False, label_column=None, warnings=[],
    )

    selected = create_plan(state, GraphClient(plan))

    assert [step.step_id for step in selected.steps] == ["linear", "visual"]


def test_analysis_step_safety_limit_is_separate_from_terminal_budget():
    state = AgentState(dataset_path="data.csv", max_experiments=2, max_analysis_steps=1)
    from agent.schemas import DatasetProfile
    state.dataset_profile = DatasetProfile(
        n_samples=30, n_features=8, numeric_features=[f"f{i}" for i in range(8)],
        categorical_features=[], boolean_features=[], missing_fraction=0, sparsity=0,
        p_to_n_ratio=8 / 30, constant_features=[], near_constant_features=[],
        has_labels=False, label_column=None, warnings=[],
    )

    with pytest.raises(ValueError, match="analysis-step slots"):
        create_plan(state, GraphClient(graph_plan("pca")))
