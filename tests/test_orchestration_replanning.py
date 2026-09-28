import json

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.io import mmwrite

from agent.llm_client import LLMClient, MockLLMClient, validate_structured_response
from agent.schemas import AnalysisPlan, ReflectionDecision
from main import run_agent
from reporting.report_generator import generate_report
from agent.state import AgentState
from agent.schemas import DatasetProfile


class ContinueMockClient(LLMClient):
    def __init__(self):
        self.mock = MockLLMClient()

    def generate_structured(self, prompt, schema):
        if schema is AnalysisPlan:
            return self.mock.generate_structured(prompt, schema)
        if schema is ReflectionDecision:
            return ReflectionDecision(
                decision="continue",
                reason="Request a genuinely new terminal comparison.",
                suggested_changes=["try method pca"],
            )
        raise AssertionError(schema)


class RepeatingClient(LLMClient):
    plan = {
        "preprocessing": [{
            "action": "standardize", "columns": None, "parameters": {},
            "rationale": "Scale once.",
        }],
        "methods": [{
            "method": "pca", "n_components": 2, "parameters": {},
            "rationale": "One terminal baseline.",
        }],
        "evaluation_metrics": ["trustworthiness"],
        "analysis_goal": "Exercise duplicate protection.",
        "rationale": "Return the same plan repeatedly.",
    }

    def generate_structured(self, prompt, schema):
        del prompt
        if schema is AnalysisPlan:
            return validate_structured_response(self.plan, schema)
        if schema is ReflectionDecision:
            return ReflectionDecision(decision="continue", reason="Try again.")
        raise AssertionError(schema)


def _write_sparse_run(tmp_path):
    matrix_path = tmp_path / "matrix.mtx"
    samples_path = tmp_path / "samples.tsv"
    features_path = tmp_path / "features.tsv"
    config_path = tmp_path / "config.yaml"
    rng = np.random.default_rng(42)
    matrix = sparse.random(
        180, 30, density=0.12, random_state=rng,
        data_rvs=lambda size: rng.integers(1, 8, size=size), format="csr",
    )
    mmwrite(matrix_path, matrix)
    samples_path.write_text("\n".join(f"s{i}" for i in range(30)) + "\n", encoding="utf-8")
    features_path.write_text(
        "\n".join(f"f{i}\tfeature_{i}" for i in range(180)) + "\n",
        encoding="utf-8",
    )
    config = {
        "random_seed": 42,
        "agent": {"max_iterations": 5, "max_experiments": 2, "max_analysis_steps": 10},
        "llm": {"provider": "mock"},
        "data": {
            "format": "matrix_market", "orientation": "features_by_samples",
            "sample_metadata_path": str(samples_path),
            "feature_metadata_path": str(features_path),
            "sample_id_column": 0, "feature_id_column": 0,
            "feature_name_column": 1, "metadata_has_header": False,
        },
        "output": {"root": str(tmp_path / "outputs")},
        "inspection": {"high_dimensional_ratio_threshold": 5, "large_p_threshold": 100},
        "evaluation": {"trustworthiness_neighbors": 3, "distance_sample_size": 30},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return matrix_path, config_path


def test_continue_reuses_preprocessing_and_intermediate_for_new_terminal(tmp_path):
    matrix_path, config_path = _write_sparse_run(tmp_path)

    state = run_agent(
        matrix_path, "synthetic_sparse", config_path, ContinueMockClient()
    )

    preprocessing_actions = [item["action"] for item in state.preprocessing_history]
    assert preprocessing_actions.count("normalize_sample_totals") == 1
    assert preprocessing_actions.count("log_transform") == 1
    assert len({item["strategy_id"] for item in state.preprocessing_history}) == 1
    intermediate = [
        item for item in state.analysis_history if item["output_role"] == "intermediate"
    ]
    terminals = [
        item for item in state.analysis_history if item["output_role"] == "terminal"
    ]
    assert len(intermediate) == 1
    assert len(terminals) == 2
    assert terminals[0]["method"] != terminals[1]["method"]
    assert terminals[1]["input_ref"] == intermediate[0]["output_ref"]
    assert state.experiments_used() == 2
    assert state.analysis_steps_used() == 3


def test_repeated_completed_terminal_plan_stops_without_noop_loop(tmp_path):
    data_path = tmp_path / "data.csv"
    config_path = tmp_path / "config.yaml"
    pd.DataFrame(np.arange(120, dtype=float).reshape(20, 6)).to_csv(data_path, index=False)
    config_path.write_text(yaml.safe_dump({
        "random_seed": 42,
        "agent": {"max_iterations": 5, "max_experiments": 3, "max_analysis_steps": 10},
        "llm": {"provider": "mock"},
        "data": {"format": "csv"},
        "output": {"root": str(tmp_path / "outputs")},
        "evaluation": {"trustworthiness_neighbors": 3, "distance_sample_size": 20},
    }), encoding="utf-8")

    state = run_agent(data_path, "duplicate_guard", config_path, RepeatingClient())

    assert len(state.analysis_history) == 1
    assert len(state.preprocessing_history) == 1
    assert state.experiments_used() == 1
    assert state.iteration < state.max_iterations
    assert any("no_new_feasible_experiment" in warning for warning in state.warnings)


def test_embedding_exports_include_aligned_sample_ids(tmp_path):
    matrix_path, config_path = _write_sparse_run(tmp_path)
    run_agent(matrix_path, "identity", config_path, ContinueMockClient())

    output = tmp_path / "outputs" / "identity" / "embeddings"
    for path in output.glob("*.csv"):
        frame = pd.read_csv(path)
        assert frame.columns[0] == "sample_id"
        assert frame["sample_id"].tolist() == [f"s{i}" for i in range(30)]
        assert all(column.startswith("component_") for column in frame.columns[1:])


def test_report_deduplicates_preprocessing_pipeline_entries(tmp_path):
    profile = DatasetProfile(
        n_samples=10, n_features=2, numeric_features=["a", "b"],
        categorical_features=[], boolean_features=[], missing_fraction=0,
        sparsity=0, p_to_n_ratio=0.2, constant_features=[],
        near_constant_features=[], has_labels=False, label_column=None, warnings=[],
    )
    state = AgentState(dataset_path="synthetic.csv", dataset_profile=profile)
    record = {
        "iteration": 0, "strategy_id": "preprocessing_same", "input_ref": "raw_X",
        "output_ref": "preprocessed_X", "action": "standardize",
        "columns": ["a", "b"], "parameters": {}, "status": "success",
        "rationale": "Scale the representation.",
    }
    state.record_preprocessing(record)
    state.record_preprocessing({**record, "iteration": 1})

    report = generate_report(state, tmp_path, "synthetic", {})
    text = report.read_text(encoding="utf-8")

    assert text.count("standardize:") == 1
    assert "`raw_X` -> `preprocessed_X`" in text
