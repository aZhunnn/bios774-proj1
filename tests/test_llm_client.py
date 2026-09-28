from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent.llm_client import (
    MockLLMClient,
    OpenAILLMClient,
    StaticLLMClient,
    validate_structured_response,
)
from agent.planner import create_plan
from agent.schemas import AnalysisPlan, DatasetProfile, ReflectionDecision
from agent.state import AgentState


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.output, output_text=None)


class FakeOpenAI:
    def __init__(self, output):
        self.responses = FakeResponses(output)


def valid_plan_response(**updates):
    response = {
        "preprocessing": [],
        "methods": [{
            "method": "pca",
            "n_components": 2,
            "parameters": {"svd_solver": "auto"},
            "rationale": "Use a linear baseline.",
        }],
        "evaluation_metrics": ["trustworthiness"],
        "analysis_goal": "Explore structure.",
        "rationale": "Use one justified method.",
    }
    response.update(updates)
    return response


def planner_state():
    return AgentState(
        dataset_path="data.csv",
        dataset_profile=DatasetProfile(
            n_samples=30,
            n_features=5,
            numeric_features=["x1", "x2", "x3", "x4", "x5"],
            categorical_features=[],
            boolean_features=[],
            missing_fraction=0.0,
            sparsity=0.0,
            p_to_n_ratio=1 / 6,
            constant_features=[],
            near_constant_features=[],
            has_labels=False,
            label_column=None,
            warnings=[],
        ),
    )


def test_validate_structured_response_accepts_dict():
    response = {
        "preprocessing": [],
        "methods": [
            {
                "method": "pca",
                "n_components": 2,
                "parameters": {},
                "rationale": "Baseline.",
            }
        ],
        "evaluation_metrics": ["trustworthiness"],
        "analysis_goal": "Explore.",
        "rationale": "Use a small plan.",
    }

    plan = validate_structured_response(response, AnalysisPlan)

    assert plan.methods[0].method == "pca"


def test_invalid_structured_response_is_rejected():
    response = {
        "preprocessing": [],
        "methods": [
            {
                "method": "not_real",
                "n_components": 2,
                "parameters": {},
                "rationale": "Invalid.",
            }
        ],
        "evaluation_metrics": ["trustworthiness"],
        "analysis_goal": "Explore.",
        "rationale": "Invalid method.",
    }

    with pytest.raises(ValidationError):
        validate_structured_response(response, AnalysisPlan)


def test_static_client_validates_response():
    client = StaticLLMClient(
        {
            "preprocessing": [],
            "methods": [],
            "evaluation_metrics": [],
            "analysis_goal": "No budget.",
            "rationale": "No experiments remain.",
        }
    )

    plan = client.generate_structured("prompt", AnalysisPlan)

    assert isinstance(plan, AnalysisPlan)


def test_mock_client_is_deterministic_for_same_prompt():
    prompt = """STATE_CONTEXT_JSON:
{"dataset_profile": {"n_samples": 20, "n_features": 4, "numeric_features": ["x"], "categorical_features": [], "constant_features": [], "missing_fraction": 0}, "remaining_experiment_budget": 2, "attempted_methods": []}
END_STATE_CONTEXT_JSON"""
    client = MockLLMClient()

    first = client.generate_structured(prompt, AnalysisPlan)
    second = client.generate_structured(prompt, AnalysisPlan)

    assert first == second


def test_openai_client_accepts_valid_structured_planner_response():
    sdk = FakeOpenAI(valid_plan_response())
    client = OpenAILLMClient(model="test-model", client=sdk)

    plan = client.generate_structured("planner prompt", AnalysisPlan)

    assert plan.methods[0].method == "pca"
    assert sdk.responses.calls[0]["text_format"] is AnalysisPlan
    assert sdk.responses.calls[0]["store"] is False


def test_openai_client_accepts_valid_structured_reflector_response():
    client = OpenAILLMClient(
        model="test-model",
        client=FakeOpenAI({
            "decision": "continue",
            "reason": "A distinct feasible experiment remains.",
            "suggested_changes": ["try method pca"],
            "preferred_embedding_id": "embedding_1",
        }),
    )

    decision = client.generate_structured("reflector prompt", ReflectionDecision)

    assert decision.decision == "continue"
    assert decision.preferred_embedding_id == "embedding_1"


@pytest.mark.parametrize(
    "response",
    ["not-json", {"decision": "maybe", "reason": "Invalid."}],
)
def test_openai_client_rejects_malformed_or_schema_invalid_response(response):
    client = OpenAILLMClient(model="test-model", client=FakeOpenAI(response))
    schema = AnalysisPlan if isinstance(response, str) else ReflectionDecision

    with pytest.raises(ValueError, match="invalid structured output"):
        client.generate_structured("prompt", schema)


def test_openai_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAILLMClient(model="test-model")


def test_openai_proposal_of_unsupported_method_is_deterministically_rejected():
    response = valid_plan_response(methods=[{
        "method": "gplvm",
        "n_components": 2,
        "parameters": {},
        "rationale": "Unsupported proposal.",
    }])
    client = OpenAILLMClient(model="test-model", client=FakeOpenAI(response))

    with pytest.raises(ValueError, match="unavailable DR method"):
        create_plan(planner_state(), client)


def test_openai_proposal_of_invalid_hyperparameters_is_deterministically_rejected():
    response = valid_plan_response(methods=[{
        "method": "isomap",
        "n_components": 2,
        "parameters": {"n_neighbors": 30},
        "rationale": "Invalid neighborhood.",
    }])
    client = OpenAILLMClient(model="test-model", client=FakeOpenAI(response))

    with pytest.raises(ValueError, match="n_neighbors"):
        create_plan(planner_state(), client)
