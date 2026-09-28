"""Command-line entry point and explicit V1 orchestration loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from agent.artifacts import AnalysisArtifact
from agent.llm_client import LLMClient, MockLLMClient, OpenAILLMClient
from agent.planner import create_plan
from agent.reflector import reflect, synthesize_completed_results
from agent.state import AgentState
from reporting.report_generator import generate_report
from tools.data_inspector import inspect_dataset
from tools.analysis_dataset import AnalysisDataset, load_analysis_dataset
from tools.dimension_reduction import run_dimension_reduction
from tools.evaluation import evaluate_embedding
from tools.preprocessing import apply_preprocessing_plan
from tools.visualization import generate_embedding_plots

LoadedDataset = AnalysisDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DeEntropy V1 workflow.")
    parser.add_argument("--data", required=True, help="Path to a configured CSV or Matrix Market dataset.")
    parser.add_argument("--name", required=True, help="Dataset run name, e.g. dataset_1.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--llm-provider",
        default=None,
        choices=["mock", "openai"],
        help="Structured-decision provider: deterministic 'mock' or configured 'openai'.",
    )
    return parser.parse_args()


def run_agent(
    dataset_path: str | Path,
    name: str,
    config_path: str | Path = "config.yaml",
    llm_client: LLMClient | None = None,
) -> AgentState:
    """Run Dataset -> Inspect -> Plan -> Execute -> Evaluate -> Reflect -> Report."""
    config = load_config(config_path)
    random_seed = int(config.get("random_seed", 42))
    llm = llm_client or initialize_llm_client(config)

    loaded = load_analysis_input(dataset_path, config.get("data", {}))
    profile = inspect_dataset(loaded, config)
    labels = loaded.labels
    if loaded.X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after excluding the label column.")

    output_paths = create_output_dirs(config, name)
    logger = setup_logging(output_paths["root"], name)
    logger.info("starting run dataset=%s name=%s", dataset_path, name)

    state = AgentState(
        dataset_path=str(dataset_path),
        dataset_profile=profile,
        input_metadata={
            **loaded.input_metadata,
            "matrix_metadata": loaded.matrix_metadata,
            "sample_ids": loaded.sample_ids.astype(str).tolist(),
            "feature_ids": loaded.feature_ids.astype(str).tolist(),
        },
        max_iterations=int(config.get("agent", {}).get("max_iterations", 5)),
        max_experiments=int(config.get("agent", {}).get("max_experiments", 10)),
        max_analysis_steps=int(config.get("agent", {}).get("max_analysis_steps", 20)),
    )
    save_json(output_paths["root"] / "dataset_profile.json", profile.model_dump())

    raw_data = loaded
    artifacts: dict[str, AnalysisArtifact] = {
        "raw_X": AnalysisArtifact(
            artifact_id="raw_X",
            data=_feature_matrix(raw_data),
            role="root",
            sample_ids=raw_data.sample_ids,
            signature="raw_X",
            n_features=raw_data.X.shape[1],
        )
    }
    artifacts_by_signature: dict[str, AnalysisArtifact] = {}
    preprocessing_cache: dict[str, AnalysisDataset] = {}
    seen_plan_signatures: set[str] = set()
    while not state.finished:
        logger.info("iteration=%s planning", state.iteration)
        plan = create_plan(state, llm)
        logger.info("iteration=%s planner_decision=%s", state.iteration, plan.model_dump())

        preprocessing_signature = _preprocessing_signature(plan.preprocessing)
        preprocessing_is_new = preprocessing_signature not in preprocessing_cache
        plan_signature = _plan_signature(plan)
        if plan_signature in seen_plan_signatures:
            state.warnings.append("no_new_feasible_experiment: repeated identical plan")
            state.finished = True
            logger.warning("iteration=%s repeated identical plan", state.iteration)
            break
        seen_plan_signatures.add(plan_signature)
        state.record_plan({
            "iteration": state.iteration,
            "signature": plan_signature,
            "preprocessing_signature": preprocessing_signature,
            "step_ids": [step.step_id for step in plan.steps],
            "terminal_step_ids": [
                step.step_id for step in plan.steps if step.output_role == "terminal"
            ],
        })

        if not plan.steps:
            reason = "no_new_feasible_experiment"
            state.warnings.append(reason)
            state.finished = True
            logger.warning("iteration=%s %s", state.iteration, reason)
            break

        if preprocessing_is_new:
            preprocessing_actions = _actions_for_feature_data(plan.preprocessing, raw_data)
            try:
                preprocessed_data, preprocessing_history = apply_preprocessing_plan(
                    raw_data,
                    preprocessing_actions,
                    random_seed=random_seed,
                )
                for action, item in zip(preprocessing_actions, preprocessing_history):
                    item = {
                        "iteration": state.iteration,
                        "strategy_id": preprocessing_signature,
                        "input_ref": "raw_X",
                        "output_ref": "preprocessed_X",
                        "rationale": action.rationale,
                        **item,
                    }
                    state.record_preprocessing(item)
                    logger.info("iteration=%s preprocessing=%s", state.iteration, item)
                preprocessing_cache[preprocessing_signature] = preprocessed_data
            except Exception as exc:
                message = f"Preprocessing failed: {exc}"
                state.errors.append(message)
                logger.exception("iteration=%s %s", state.iteration, message)
                state.finished = True
                break
        else:
            preprocessed_data = preprocessing_cache[preprocessing_signature]
            logger.info(
                "iteration=%s reusing preprocessing strategy=%s",
                state.iteration,
                preprocessing_signature,
            )

        X = _feature_matrix(preprocessed_data)
        if X.shape[1] == 0:
            raise ValueError("No usable numeric features are available after preprocessing.")

        preprocessed_artifact = AnalysisArtifact(
            artifact_id="preprocessed_X",
            data=X,
            role="root",
            parent_ref="raw_X",
            sample_ids=preprocessed_data.sample_ids,
            signature=preprocessing_signature,
            n_features=X.shape[1],
        )
        artifacts["preprocessed_X"] = preprocessed_artifact
        if not any(
            item.get("artifact_id") == "preprocessed_X"
            and item.get("signature") == preprocessing_signature
            for item in state.artifact_history
        ):
            state.record_artifact({
                "artifact_id": "preprocessed_X",
                "signature": preprocessing_signature,
                "role": "root",
                "parent_ref": "raw_X",
                "preprocessing_strategy_id": preprocessing_signature,
                "success": True,
                "profile": _artifact_profile(X),
            })

        iteration_embeddings = []
        failed_artifacts: dict[str, str] = {}
        new_analysis_steps = 0
        new_terminal_experiments = 0
        for step in plan.steps:
            if state.analysis_step_limit_exhausted():
                state.finished = True
                logger.info("analysis-step limit exhausted before step=%s", step.step_id)
                break
            if step.output_role == "terminal" and state.experiment_budget_exhausted():
                state.finished = True
                logger.info("terminal experiment budget exhausted before step=%s", step.step_id)
                break

            if step.input_ref not in artifacts:
                parent_error = failed_artifacts.get(
                    step.input_ref, f"Input artifact '{step.input_ref}' is unavailable."
                )
                message = f"Blocked by failed dependency: {parent_error}"
                failed_artifacts[step.output_ref] = message
                analysis_record = {
                    "timestamp": utc_now(),
                    "iteration": state.iteration,
                    "step_id": step.step_id,
                    "method": step.method,
                    "input_ref": step.input_ref,
                    "output_ref": step.output_ref,
                    "output_role": step.output_role,
                    "success": False,
                    "execution_status": "blocked_dependency",
                    "error_message": message,
                    "counts_toward_budget": False,
                    "counts_toward_experiment_budget": False,
                    "counts_toward_step_limit": False,
                    "rationale": step.rationale,
                }
                state.record_analysis(analysis_record)
                logger.warning("iteration=%s blocked_step=%s", state.iteration, analysis_record)
                continue

            parent = artifacts[step.input_ref]
            artifact_signature = _analysis_artifact_signature(
                parent.signature,
                step.method,
                step.n_components,
                step.parameters,
                random_seed,
            )
            reusable = artifacts_by_signature.get(artifact_signature)
            if reusable is not None:
                artifacts[step.output_ref] = reusable
                logger.info(
                    "iteration=%s reusing artifact=%s for step=%s",
                    state.iteration,
                    reusable.artifact_id,
                    step.step_id,
                )
                continue

            result = run_dimension_reduction(
                parent.data,
                method=step.method,
                n_components=step.n_components,
                parameters=step.parameters,
                random_seed=random_seed,
            )
            embedding_path = None
            if result.success:
                embedding_path = save_embedding_csv(
                    result,
                    output_paths["embeddings"],
                    parent.sample_ids,
                )
                produced_artifact = AnalysisArtifact(
                    artifact_id=step.output_ref,
                    data=result.embedding,
                    role=step.output_role,
                    source_step_id=step.step_id,
                    method=step.method,
                    parent_ref=step.input_ref,
                    sample_ids=parent.sample_ids,
                    signature=artifact_signature,
                    n_features=step.n_components,
                )
                artifacts[step.output_ref] = produced_artifact
                artifacts_by_signature[artifact_signature] = produced_artifact
                state.record_artifact({
                    "artifact_id": step.output_ref,
                    "signature": artifact_signature,
                    "role": step.output_role,
                    "source_step_id": step.step_id,
                    "method": step.method,
                    "parent_ref": step.input_ref,
                    "success": True,
                    "profile": _artifact_profile(result.embedding),
                })
                if step.output_role == "terminal":
                    iteration_embeddings.append(result)
            else:
                failed_artifacts[step.output_ref] = result.error_message or "Unknown parent failure."

            analysis_record = {
                "timestamp": utc_now(),
                "iteration": state.iteration,
                "embedding_id": result.embedding_id,
                "method": result.method,
                "parameters": result.parameters,
                "planned_parameters": step.parameters,
                "n_components": result.n_components,
                "runtime_seconds": result.runtime_seconds,
                "success": result.success,
                "error_message": result.error_message,
                "metadata": result.metadata,
                "step_id": step.step_id,
                "input_ref": step.input_ref,
                "output_ref": step.output_ref,
                "output_role": step.output_role,
                "execution_status": "completed" if result.success else "failed",
                "experiment_id": step.output_ref if step.output_role == "terminal" else None,
                "counts_toward_budget": step.output_role == "terminal",
                "counts_toward_experiment_budget": step.output_role == "terminal",
                "counts_toward_step_limit": True,
                "rationale": step.rationale,
                "embedding_path": str(embedding_path) if embedding_path else None,
                "provenance": {
                    "root_ref": "preprocessed_X",
                    "parent_ref": step.input_ref,
                    "output_ref": step.output_ref,
                    "parent_artifact_role": parent.role,
                },
            }
            state.record_analysis(analysis_record)
            new_analysis_steps += 1
            if step.output_role == "terminal":
                new_terminal_experiments += 1
            logger.info("iteration=%s dr_result=%s", state.iteration, analysis_record)

            if not result.success:
                state.warnings.append(
                    f"{result.method} failed: {result.error_message}"
                )
                continue

            if step.output_role != "terminal":
                continue

            evaluation = evaluate_embedding(parent.data, result, labels=labels, config=config)
            evaluation_record = {
                "timestamp": utc_now(),
                "iteration": state.iteration,
                **evaluation.model_dump(),
                "step_id": step.step_id,
                "input_ref": step.input_ref,
                "output_ref": step.output_ref,
                "provenance": {
                    "parent_fidelity_reference": step.input_ref,
                    "root_fidelity_reference": "preprocessed_X",
                },
                "parent_fidelity": evaluation.model_dump(),
                "root_fidelity": _root_fidelity_record(
                    X, step.input_ref, result, labels, config
                ),
            }
            state.record_evaluation(evaluation_record)
            logger.info("iteration=%s evaluation=%s", state.iteration, evaluation_record)

        plot_paths = generate_embedding_plots(
            iteration_embeddings,
            output_paths["plots"],
            labels=labels,
        )
        for plot_path in plot_paths:
            logger.info("iteration=%s plot=%s", state.iteration, plot_path)

        if new_analysis_steps == 0 and new_terminal_experiments == 0:
            state.warnings.append("no_new_feasible_experiment: iteration produced no new evidence")
            state.finished = True
            logger.warning("iteration=%s no-op iteration blocked", state.iteration)
            persist_run_outputs(state, output_paths)
            break

        persist_run_outputs(state, output_paths)
        decision = reflect(state, llm)
        logger.info("iteration=%s reflection=%s", state.iteration, decision.model_dump())

        if decision.decision == "stop":
            state.finished = True
        else:
            state.advance_iteration()
            state.apply_stop_conditions()

        persist_run_outputs(state, output_paths)

    state.record_final_synthesis(synthesize_completed_results(state))
    report_path = generate_report(
        state=state,
        output_dir=output_paths["root"],
        dataset_name=name,
        config=config,
    )
    logger.info("report=%s", report_path)
    persist_run_outputs(state, output_paths)
    return state


def load_config(config_path: str | Path) -> dict[str, Any]:
    try:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise ValueError(f"Unable to read config file: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Phase 4 supports CSV input only.")
    try:
        return pd.read_csv(path)
    except OSError as exc:
        raise ValueError(f"Unable to read dataset: {dataset_path}") from exc


def load_analysis_input(
    dataset_path: str | Path,
    data_config: dict[str, Any] | None = None,
) -> LoadedDataset:
    """Load any configured input into the shared AnalysisDataset abstraction."""
    return load_analysis_dataset(dataset_path, data_config)


def _validate_id_column(df: pd.DataFrame, id_column: str, source: str) -> None:
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' was not found in {source}.")
    if df[id_column].isna().any():
        raise ValueError(f"ID column '{id_column}' contains missing values in {source}.")
    duplicates = df.loc[df[id_column].duplicated(), id_column]
    if not duplicates.empty:
        raise ValueError(
            f"Duplicate IDs found in {source} for column '{id_column}': "
            + ", ".join(duplicates.astype(str).unique()[:5])
        )


def initialize_llm_client(config: dict[str, Any]) -> LLMClient:
    llm_config = config.get("llm", {})
    provider = str(llm_config.get("provider", "mock")).lower()
    if provider in {"mock", "configurable"}:
        return MockLLMClient()
    if provider == "openai":
        return OpenAILLMClient(
            model=str(llm_config.get("model", "")),
            temperature=float(llm_config.get("temperature", 0)),
            api_key_env=str(llm_config.get("api_key_env", "OPENAI_API_KEY")),
        )
    raise ValueError(
        f"Unsupported LLM provider: {provider}. Use provider='mock' or 'openai'."
    )


def create_output_dirs(config: dict[str, Any], dataset_name: str) -> dict[str, Path]:
    root = Path(config.get("output", {}).get("root", "outputs")) / dataset_name
    paths = {
        "root": root,
        "embeddings": root / "embeddings",
        "metrics": root / "metrics",
        "plots": root / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(output_dir: Path, dataset_name: str) -> logging.Logger:
    logger = logging.getLogger(f"deentropy.{dataset_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path = output_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def persist_run_outputs(state: AgentState, output_paths: dict[str, Path]) -> None:
    save_json(output_paths["root"] / "analysis_history.json", state.analysis_history)
    save_json(
        output_paths["metrics"] / "evaluation_results.json",
        state.evaluation_history,
    )
    save_json(output_paths["root"] / "final_state.json", state.model_dump())


def save_embedding_csv(
    result: Any,
    output_dir: Path,
    sample_ids: Any,
) -> Path:
    embedding = np.asarray(result.embedding, dtype=float)
    ids = pd.Index(sample_ids)
    if len(ids) != embedding.shape[0]:
        raise ValueError(
            "Embedding row count does not match sample provenance: "
            f"rows={embedding.shape[0]}, sample_ids={len(ids)}."
        )
    if ids.has_duplicates:
        raise ValueError("Embedding sample provenance contains duplicate sample IDs.")
    columns = [f"component_{index + 1}" for index in range(embedding.shape[1])]
    path = output_dir / f"{result.method}.csv"
    if path.exists():
        path = output_dir / f"{result.embedding_id}.csv"
    frame = pd.DataFrame(embedding, columns=columns)
    frame.insert(0, "sample_id", ids.astype(str).to_numpy())
    frame.to_csv(path, index=False)
    return path


def _preprocessing_signature(actions: list[Any]) -> str:
    payload = [
        {
            "action": action.action,
            "columns": action.columns,
            "parameters": action.parameters,
        }
        for action in actions
    ]
    return _stable_digest(payload, prefix="preprocessing")


def _plan_signature(plan: Any) -> str:
    return _stable_digest(
        {
            "preprocessing": [
                {
                    "action": action.action,
                    "columns": action.columns,
                    "parameters": action.parameters,
                }
                for action in plan.preprocessing
            ],
            "steps": [
                {
                    "method": step.method,
                    "input_ref": step.input_ref,
                    "output_ref": step.output_ref,
                    "output_role": step.output_role,
                    "n_components": step.n_components,
                    "parameters": step.parameters,
                }
                for step in plan.steps
            ],
        },
        prefix="plan",
    )


def _analysis_artifact_signature(
    parent_signature: str | None,
    method: str,
    n_components: int,
    parameters: dict[str, Any],
    random_seed: int,
) -> str:
    return _stable_digest(
        {
            "parent_signature": parent_signature,
            "method": method,
            "n_components": n_components,
            "parameters": parameters,
            "random_seed": random_seed,
        },
        prefix="artifact",
    )


def _stable_digest(payload: Any, prefix: str) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]}"


def _artifact_profile(data: Any) -> dict[str, Any]:
    n_samples, n_features = data.shape
    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "p_to_n_ratio": float(n_features / n_samples) if n_samples else 0.0,
        "storage_type": "sparse" if sparse.issparse(data) else "dense",
    }


def _root_fidelity_record(
    root_X: Any,
    input_ref: str,
    result: Any,
    labels: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate against root only when practical, preserving reference-space identity."""
    if input_ref == "preprocessed_X":
        return {
            "status": "same_as_parent",
            "reference_ref": "preprocessed_X",
        }
    if sparse.issparse(root_X):
        return {
            "status": "not_computed",
            "reference_ref": "preprocessed_X",
            "reason": "Root-fidelity metrics currently require dense input; the sparse root was not densified.",
        }
    root_evaluation = evaluate_embedding(root_X, result, labels=labels, config=config)
    return {
        "status": "computed",
        "reference_ref": "preprocessed_X",
        "evaluation": root_evaluation.model_dump(),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def _extract_labels(df: pd.DataFrame, label_column: str | None) -> pd.Series | None:
    if label_column and label_column in df.columns:
        return df[label_column]
    return None


def _feature_dataframe(df: pd.DataFrame, label_column: str | None) -> pd.DataFrame:
    if label_column and label_column in df.columns:
        return df.drop(columns=[label_column])
    return df.copy()


def _feature_matrix(dataset: AnalysisDataset) -> Any:
    if dataset.is_sparse:
        return dataset.X
    return dataset.X.select_dtypes(include=[np.number, "bool"])


def _actions_for_feature_data(
    actions: list[Any], dataset: AnalysisDataset
) -> list[Any]:
    filtered = []
    available_columns = {str(item) for item in dataset.feature_ids}
    for action in actions:
        if action.columns is None:
            filtered.append(action)
            continue
        columns = [column for column in action.columns if column in available_columns]
        if columns:
            filtered.append(action.model_copy(update={"columns": columns}))
    return filtered


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.llm_provider:
        config.setdefault("llm", {})["provider"] = args.llm_provider
    run_agent(
        dataset_path=args.data,
        name=args.name,
        config_path=args.config,
        llm_client=initialize_llm_client(config),
    )


if __name__ == "__main__":
    main()
