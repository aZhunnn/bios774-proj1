from agent.schemas import ReflectionDecision
from agent.state import AgentState


def test_state_iteration_stop_condition():
    state = AgentState(dataset_path="data.csv", max_iterations=1)

    state.advance_iteration()

    assert state.iteration == 1
    assert state.finished is True


def test_state_records_history_and_best_embedding():
    state = AgentState(dataset_path="data.csv")
    decision = ReflectionDecision(
        decision="stop",
        reason="Good enough.",
        preferred_embedding_id="pca_1",
    )

    state.record_preprocessing({"action": "standardize", "status": "success"})
    state.record_analysis({"embedding_id": "pca_1"})
    state.record_evaluation({"embedding_id": "pca_1", "metrics": {"trustworthiness": 0.9}})
    state.record_reflection(decision)

    assert state.preprocessing_history
    assert state.analysis_history
    assert state.evaluation_history
    assert state.reflection_history
    assert state.best_embedding_id == "pca_1"


def test_state_experiment_budget_stop_condition():
    state = AgentState(dataset_path="data.csv", max_experiments=1)
    state.record_analysis({"embedding_id": "pca_1"})

    state.apply_stop_conditions()

    assert state.finished is True

