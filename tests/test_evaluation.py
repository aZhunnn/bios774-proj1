import numpy as np
from sklearn.metrics import silhouette_score

from agent.schemas import EmbeddingResult
from tools.dimension_reduction import run_dimension_reduction
from tools.evaluation import (
    distance_correlations,
    evaluate_embedding,
    knn_preservation,
    neighborhood_label_agreement,
)


def make_matrix(n_samples=30, n_features=5):
    rng = np.random.default_rng(321)
    return rng.normal(size=(n_samples, n_features))


def test_evaluate_embedding_computes_required_metrics():
    X = make_matrix()
    embedding = run_dimension_reduction(X, "pca", 2, {}, random_seed=42)

    evaluation = evaluate_embedding(
        X,
        embedding,
        config={"evaluation": {"trustworthiness_neighbors": 5, "distance_sample_size": 30}},
    )

    assert evaluation.embedding_id == embedding.embedding_id
    assert evaluation.metrics["trustworthiness"] is not None
    assert evaluation.metrics["knn_preservation"] is not None
    assert evaluation.metrics["distance_pearson_correlation"] is not None
    assert evaluation.metrics["distance_spearman_correlation"] is not None
    assert 0 <= evaluation.metrics["trustworthiness"] <= 1
    assert 0 <= evaluation.metrics["knn_preservation"] <= 1
    assert evaluation.intrinsic_metrics == evaluation.metrics
    assert evaluation.external_validation_metrics == {}


def test_knn_preservation_is_high_for_identical_embedding():
    X = make_matrix(n_samples=20, n_features=3)

    score = knn_preservation(X, X, n_neighbors=4)

    assert score == 1.0


def test_distance_correlations_are_high_for_identical_embedding():
    X = make_matrix(n_samples=20, n_features=3)

    pearson, spearman = distance_correlations(X, X, sample_size=20, random_seed=42)

    assert pearson is not None
    assert spearman is not None
    assert pearson > 0.99
    assert spearman > 0.99


def test_evaluate_failed_embedding_returns_warnings_and_empty_metrics():
    failed = EmbeddingResult(
        embedding_id="bad",
        method="umap",
        parameters={},
        n_components=2,
        embedding=None,
        runtime_seconds=0.01,
        success=False,
        error_message="failure",
        metadata={},
    )

    evaluation = evaluate_embedding(make_matrix(), failed)

    assert evaluation.metrics["trustworthiness"] is None
    assert evaluation.metrics["knn_preservation"] is None
    assert evaluation.warnings


def test_umap_evaluation_notes_global_distance_caution():
    X = make_matrix()
    embedding = run_dimension_reduction(
        X,
        "umap",
        2,
        {"n_neighbors": 5, "n_epochs": 20},
        random_seed=42,
    )

    evaluation = evaluate_embedding(X, embedding)

    assert embedding.success is True
    assert any("global distance" in note for note in evaluation.notes)


def test_external_validation_metrics_are_computed_and_separated():
    X = np.array(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]]
    )
    labels = np.array(["a", "a", "b", "b"])
    embedding = EmbeddingResult(
        embedding_id="separated",
        method="pca",
        parameters={},
        n_components=2,
        embedding=X,
        runtime_seconds=0.01,
        success=True,
        metadata={},
    )

    evaluation = evaluate_embedding(
        X,
        embedding,
        labels=labels,
        config={"evaluation": {"trustworthiness_neighbors": 1, "external_validation_neighbors": 1}},
    )

    external = evaluation.external_validation_metrics
    assert external["silhouette_score"] == silhouette_score(X, labels)
    assert external["neighborhood_label_agreement"] == 1.0
    assert "silhouette_score" not in evaluation.intrinsic_metrics
    assert "trustworthiness" not in external


def test_neighborhood_label_agreement_matches_known_neighbor_structure():
    embedding = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels = np.array([0, 0, 1, 1])

    assert neighborhood_label_agreement(embedding, labels, n_neighbors=1) == 1.0


def test_single_label_class_is_handled_gracefully():
    X = make_matrix(n_samples=8, n_features=3)
    embedding = run_dimension_reduction(X, "pca", 2, {}, random_seed=42)

    evaluation = evaluate_embedding(X, embedding, labels=["one"] * len(X))

    assert evaluation.external_validation_metrics["silhouette_score"] is None
    assert evaluation.external_validation_metrics["neighborhood_label_agreement"] is None
    assert any("at least two" in warning for warning in evaluation.warnings)


def test_unlabeled_evaluation_continues_without_external_metrics():
    X = make_matrix(n_samples=12, n_features=4)
    embedding = run_dimension_reduction(X, "pca", 2, {}, random_seed=42)

    evaluation = evaluate_embedding(X, embedding, labels=None)

    assert evaluation.intrinsic_metrics["trustworthiness"] is not None
    assert evaluation.external_validation_metrics == {}
