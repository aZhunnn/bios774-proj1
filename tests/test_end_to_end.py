import json
import subprocess
import sys

import pandas as pd
import yaml

from agent.llm_client import LLMClient, validate_structured_response
from agent.schemas import AnalysisPlan, ReflectionDecision
from main import run_agent


def write_config(path, output_root, max_iterations=2, max_experiments=2):
    config = {
        "random_seed": 42,
        "agent": {
            "max_iterations": max_iterations,
            "max_experiments": max_experiments,
        },
        "llm": {"provider": "mock", "model": "mock", "temperature": 0},
        "data": {"label_column": "label"},
        "output": {"root": str(output_root)},
        "inspection": {
            "near_constant_threshold": 0.99,
            "high_cardinality_threshold": 50,
            "small_n_threshold": 10,
            "large_p_threshold": 1000,
            "scaling_ratio_threshold": 100,
        },
        "evaluation": {
            "trustworthiness_neighbors": 5,
            "distance_sample_size": 40,
            "stability_runs": 1,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def write_dataset(path, n_samples=32):
    rows = []
    for index in range(n_samples):
        label = "a" if index < n_samples / 2 else "b"
        rows.append(
            {
                "x1": index,
                "x2": index * 0.5,
                "x3": (index % 5) * 1.5,
                "x4": 1 if index % 2 else 0,
                "constant": 7,
                "label": label,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_full_v1_pipeline_with_mock_llm_produces_required_outputs(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "outputs"
    write_dataset(dataset_path)
    write_config(config_path, output_root, max_iterations=2, max_experiments=2)

    state = run_agent(dataset_path, "dataset_1", config_path)
    output_dir = output_root / "dataset_1"

    assert state.finished is True
    assert state.dataset_profile is not None
    assert state.preprocessing_history
    assert state.evaluation_history
    assert state.reflection_history

    analysis_history = json.loads(
        (output_dir / "analysis_history.json").read_text(encoding="utf-8")
    )
    methods = {item["method"] for item in analysis_history}
    assert methods == {"mds", "kernel_pca"}
    assert any(item["success"] for item in analysis_history)

    metrics = json.loads(
        (output_dir / "metrics" / "evaluation_results.json").read_text(encoding="utf-8")
    )
    assert metrics
    assert all("trustworthiness" in item["metrics"] for item in metrics)
    assert all("knn_preservation" in item["metrics"] for item in metrics)
    assert all("distance_pearson_correlation" in item["metrics"] for item in metrics)
    assert all(item["intrinsic_metrics"] == item["metrics"] for item in metrics)
    assert all("silhouette_score" in item["external_validation_metrics"] for item in metrics)
    assert state.final_recommendation is not None
    assert state.final_recommendation.primary_embedding_id is not None

    assert (output_dir / "dataset_profile.json").exists()
    assert (output_dir / "final_state.json").exists()
    assert (output_dir / "generated_report_1.md").exists()
    assert (output_dir / "embeddings" / "mds.csv").exists()
    assert (output_dir / "embeddings" / "kernel_pca.csv").exists()
    assert (output_dir / "plots" / "mds_embedding.png").exists()
    assert (output_dir / "plots" / "kernel_pca_embedding.png").exists()
    assert len(analysis_history) <= 2


def test_cli_runs_with_mock_llm_and_csv_input(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "outputs"
    write_dataset(dataset_path, n_samples=20)
    write_config(config_path, output_root, max_iterations=1, max_experiments=1)

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--data",
            str(dataset_path),
            "--name",
            "dataset_2",
            "--config",
            str(config_path),
            "--llm-provider",
            "mock",
        ],
        check=False,
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output_dir = output_root / "dataset_2"
    assert (output_dir / "generated_report_2.md").exists()
    assert (output_dir / "analysis_history.json").exists()


class FailureThenSuccessClient(LLMClient):
    def generate_structured(self, prompt, schema):
        del prompt
        if schema is AnalysisPlan:
            return validate_structured_response(
                {
                    "preprocessing": [
                        {
                            "action": "standardize",
                            "columns": None,
                            "parameters": {},
                            "rationale": "Scale numeric features.",
                        }
                    ],
                    "methods": [
                        {
                            "method": "pca",
                            "n_components": 2,
                            "parameters": {},
                            "rationale": "Intentional failure for resilience test.",
                        },
                        {
                            "method": "kernel_pca",
                            "n_components": 2,
                            "parameters": {"kernel": "linear"},
                            "rationale": "Valid fallback after a failed experiment.",
                        },
                    ],
                    "evaluation_metrics": ["trustworthiness", "knn_preservation"],
                    "analysis_goal": "Verify graceful experiment failure handling.",
                    "rationale": "One planned experiment should fail and one should succeed.",
                },
                schema,
            )
        if schema is ReflectionDecision:
            return validate_structured_response(
                {
                    "decision": "stop",
                    "reason": "Resilience test complete.",
                    "suggested_changes": [],
                    "preferred_embedding_id": None,
                },
                schema,
            )
        raise ValueError(schema)


def test_failed_dr_experiment_is_recorded_without_crashing_workflow(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.csv"
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "outputs"
    write_dataset(dataset_path, n_samples=18)
    write_config(config_path, output_root, max_iterations=1, max_experiments=2)

    import main as main_module

    real_runner = main_module.run_dimension_reduction
    call_count = 0

    def fail_first_experiment(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            result = real_runner(*args, **kwargs)
            return result.model_copy(
                update={"success": False, "embedding": None, "error_message": "injected failure"}
            )
        return real_runner(*args, **kwargs)

    monkeypatch.setattr(main_module, "run_dimension_reduction", fail_first_experiment)
    state = run_agent(
        dataset_path,
        "failure_case",
        config_path,
        llm_client=FailureThenSuccessClient(),
    )
    output_dir = output_root / "failure_case"
    history = json.loads(
        (output_dir / "analysis_history.json").read_text(encoding="utf-8")
    )

    assert state.finished is True
    assert len(history) == 2
    assert history[0]["success"] is False
    assert history[0]["error_message"]
    assert history[1]["success"] is True
    assert state.evaluation_history
    assert (output_dir / "generated_report.md").exists()
