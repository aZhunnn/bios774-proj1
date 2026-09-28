import json

import pandas as pd
import yaml

from main import load_dataset, run_agent


def write_config(path, max_iterations=1, max_experiments=1):
    config = {
        "random_seed": 42,
        "agent": {
            "max_iterations": max_iterations,
            "max_experiments": max_experiments,
        },
        "llm": {"provider": "mock", "model": "mock", "temperature": 0},
        "data": {"label_column": None},
        "output": {"root": str(path.parent / "outputs")},
        "inspection": {
            "near_constant_threshold": 0.99,
            "high_cardinality_threshold": 50,
            "small_n_threshold": 10,
            "large_p_threshold": 1000,
            "scaling_ratio_threshold": 100,
        },
        "evaluation": {
            "trustworthiness_neighbors": 5,
            "distance_sample_size": 50,
            "stability_runs": 1,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_load_dataset_supports_csv(tmp_path):
    dataset_path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(dataset_path, index=False)

    df = load_dataset(dataset_path)

    assert df.shape == (2, 2)


def test_run_agent_with_mock_client_generates_phase4_outputs(tmp_path):
    dataset_path = tmp_path / "data.csv"
    config_path = tmp_path / "config.yaml"
    pd.DataFrame(
        {
            "x1": range(20),
            "x2": [value * 2 for value in range(20)],
            "x3": [value % 3 for value in range(20)],
        }
    ).to_csv(dataset_path, index=False)
    write_config(config_path)

    state = run_agent(dataset_path, "dataset_1", config_path)
    output_dir = tmp_path / "outputs" / "dataset_1"

    assert state.finished is True
    assert (output_dir / "dataset_profile.json").exists()
    assert (output_dir / "analysis_history.json").exists()
    assert (output_dir / "final_state.json").exists()
    assert (output_dir / "metrics" / "evaluation_results.json").exists()
    assert (output_dir / "embeddings" / "mds.csv").exists()
    assert (output_dir / "plots" / "mds_embedding.png").exists()
    assert (output_dir / "generated_report_1.md").exists()
    assert (output_dir / "run.log").exists()

    history = json.loads((output_dir / "analysis_history.json").read_text(encoding="utf-8"))
    assert history[0]["method"] == "mds"
    assert history[0]["success"] is True
