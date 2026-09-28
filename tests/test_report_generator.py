from agent.schemas import DatasetProfile
from agent.state import AgentState
from reporting.report_generator import generate_report, report_filename


def profile():
    return DatasetProfile(
        n_samples=12,
        n_features=3,
        numeric_features=["x1", "x2"],
        categorical_features=["group"],
        boolean_features=[],
        missing_fraction=0.0,
        missingness_by_column={},
        sparsity=0.0,
        p_to_n_ratio=0.25,
        constant_features=[],
        near_constant_features=[],
        duplicate_rows=0,
        numeric_summaries={},
        potential_scaling_issues=[],
        high_cardinality_categorical_features=[],
        has_labels=True,
        label_column="group",
        warnings=[],
    )


def test_report_filename_for_numbered_datasets():
    assert report_filename("dataset_1") == "generated_report_1.md"
    assert report_filename("dataset_2") == "generated_report_2.md"
    assert report_filename("custom") == "generated_report.md"


def test_generate_report_uses_recorded_state(tmp_path):
    state = AgentState(dataset_path="data.csv", dataset_profile=profile())
    state.record_preprocessing(
        {
            "iteration": 0,
            "action": "standardize",
            "columns": ["x1", "x2"],
            "status": "success",
        }
    )
    state.record_analysis(
        {
            "iteration": 0,
            "method": "pca",
            "embedding_id": "pca_1",
            "parameters": {"svd_solver": "auto"},
            "success": True,
            "rationale": "Baseline.",
        }
    )
    state.record_evaluation(
        {
            "iteration": 0,
            "embedding_id": "pca_1",
            "intrinsic_metrics": {"trustworthiness": 0.91},
            "external_validation_metrics": {
                "silhouette_score": 0.72,
                "neighborhood_label_agreement": 0.8,
            },
            "metrics": {"trustworthiness": 0.91},
            "warnings": [],
            "notes": [],
        }
    )

    path = generate_report(state, tmp_path, "dataset_1", {"random_seed": 42})
    text = path.read_text(encoding="utf-8")

    assert path.name == "generated_report_1.md"
    assert "Samples: 12" in text
    assert "trustworthiness: 0.91" in text
    assert "Intrinsic Embedding Evaluation" in text
    assert "External Label-Based Validation" in text
    assert "silhouette_score: 0.72" in text
    assert "labels were used only for post-hoc validation" in text.lower()
    assert "Final Recommendation" in text
    assert "pca_1" in text
