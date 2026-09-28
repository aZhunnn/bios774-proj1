"""Single source of truth for a DeEntropy analysis run."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.schemas import AnalysisPlan, DatasetProfile, FinalSynthesis, ReflectionDecision


class AgentState(BaseModel):
    dataset_path: str

    dataset_profile: DatasetProfile | None = None
    input_metadata: dict[str, Any] = Field(default_factory=dict)

    preprocessing_history: list[dict[str, Any]] = Field(default_factory=list)
    analysis_history: list[dict[str, Any]] = Field(default_factory=list)
    evaluation_history: list[dict[str, Any]] = Field(default_factory=list)
    reflection_history: list[dict[str, Any]] = Field(default_factory=list)
    artifact_history: list[dict[str, Any]] = Field(default_factory=list)
    plan_history: list[dict[str, Any]] = Field(default_factory=list)

    current_plan: AnalysisPlan | None = None

    best_embedding_id: str | None = None
    final_recommendation: FinalSynthesis | None = None

    iteration: int = 0
    max_iterations: int = 5
    max_experiments: int = 10
    max_analysis_steps: int = 20
    finished: bool = False

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def record_preprocessing(self, metadata: dict[str, Any]) -> None:
        self.preprocessing_history.append(metadata)

    def record_analysis(self, metadata: dict[str, Any]) -> None:
        self.analysis_history.append(metadata)

    def record_evaluation(self, metadata: dict[str, Any]) -> None:
        self.evaluation_history.append(metadata)

    def record_reflection(self, decision: ReflectionDecision) -> None:
        self.reflection_history.append(decision.model_dump())
        if decision.preferred_embedding_id:
            self.best_embedding_id = decision.preferred_embedding_id

    def record_artifact(self, metadata: dict[str, Any]) -> None:
        self.artifact_history.append(metadata)

    def record_plan(self, metadata: dict[str, Any]) -> None:
        self.plan_history.append(metadata)

    def record_final_synthesis(self, synthesis: FinalSynthesis) -> None:
        self.final_recommendation = synthesis
        self.best_embedding_id = synthesis.primary_embedding_id

    def advance_iteration(self) -> None:
        self.iteration += 1
        if self.iteration >= self.max_iterations:
            self.finished = True

    def experiment_budget_exhausted(self) -> bool:
        return self.experiments_used() >= self.max_experiments

    def experiments_used(self) -> int:
        return sum(
            1
            for item in self.analysis_history
            if item.get(
                "counts_toward_experiment_budget",
                item.get("output_role", "terminal") == "terminal"
                and item.get("execution_status") != "blocked_dependency",
            )
        )

    def analysis_steps_used(self) -> int:
        return sum(
            1
            for item in self.analysis_history
            if item.get("counts_toward_step_limit", item.get("execution_status") != "blocked_dependency")
        )

    def analysis_step_limit_exhausted(self) -> bool:
        return self.analysis_steps_used() >= self.max_analysis_steps

    def apply_stop_conditions(self) -> None:
        if (
            self.iteration >= self.max_iterations
            or self.experiment_budget_exhausted()
            or self.analysis_step_limit_exhausted()
        ):
            self.finished = True
