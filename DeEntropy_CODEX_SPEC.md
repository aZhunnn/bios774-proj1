# DeEntropy — Codex Implementation Specification

## 0. Purpose of This Document

This document is the implementation specification for **DeEntropy**, an AI agent for automated exploratory data analysis (EDA) through dimension reduction.

Use this file as the primary coding specification for **Version 1 (V1)**.

The goal of V1 is to build a complete, reliable, reproducible, end-to-end agentic workflow that can accept a previously unseen dataset and, with minimal manual intervention:

1. inspect the dataset;
2. recommend and perform preprocessing;
3. choose suitable dimension-reduction methods;
4. choose initial hyperparameters;
5. execute selected methods;
6. generate low-dimensional representations;
7. evaluate the embeddings;
8. decide whether another analysis iteration is needed;
9. generate visualizations;
10. produce a final analysis report.

The project is **not** intended to blindly run every available algorithm. The system should make explicit analysis decisions based on the dataset and explain those decisions.

---

# 1. System Name

## DeEntropy

**DeEntropy** is an agentic statistical analysis system for automated dimension reduction and exploratory data analysis.

The name reflects the system's purpose: reduce complexity in high-dimensional data while making the analysis process more structured, interpretable, and autonomous.

---

# 2. V1 Scope

## V1 MUST implement

The first version must implement this workflow:

\[
\boxed{
\text{Dataset}
\rightarrow
\text{Inspect}
\rightarrow
\text{Plan}
\rightarrow
\text{Execute}
\rightarrow
\text{Evaluate}
\rightarrow
\text{Reflect}
\rightarrow
\text{Report}
}
\]

The core architecture is:

```text
                         New Dataset
                              |
                              v
                     +----------------+
                     | Data Inspector |
                     | Python / Rules |
                     +-------+--------+
                             |
                             v
                       Dataset Profile
                             |
                             v
                     +----------------+
                     |  LLM Planner   |
                     +-------+--------+
                             |
                             v
                       Analysis Plan
                             |
                             v
                  +----------------------+
                  |  Statistical Tools   |
                  | preprocessing + DR   |
                  +----------+-----------+
                             |
                             v
                        Evaluation
                             |
                             v
                     +----------------+
                     | LLM Reflector  |
                     +-------+--------+
                             |
                       Good enough?
                       /          \
                     No            Yes
                     |              |
                     v              v
                 Revise Plan    Final Report
```

---

# 3. V1 MUST NOT Implement

Do **not** implement the following in V1:

- multi-agent communication;
- planner/critic multi-agent architecture;
- RAG;
- vector databases;
- document retrieval;
- Bayesian sequential method selection;
- Bayesian latent-variable models;
- Bayesian optimization;
- reinforcement learning;
- autonomous code generation by the LLM;
- unrestricted Python execution by the LLM;
- distributed execution;
- web search;
- external data acquisition.

Do not create fake placeholder implementations for these features.

The codebase should be designed so that these extensions can be added later without rewriting the core workflow.

---

# 4. Design Philosophy

DeEntropy should follow this principle:

\[
\boxed{
\text{Deterministic tools for computation}
+
\text{LLM for higher-level reasoning}
}
\]

The LLM should **not** directly manipulate dataframes, calculate statistics, or write arbitrary analysis code during execution.

Instead:

\[
\boxed{
\text{LLM}
\rightarrow
\text{structured decision}
\rightarrow
\text{approved Python tool}
\rightarrow
\text{result}
}
\]

The statistical tools must perform all numerical computation.

This separation is critical for:

- reproducibility;
- robustness;
- testability;
- transparent debugging;
- statistical correctness.

---

# 5. Technology Decisions for V1

Use the following defaults unless implementation constraints make a small change necessary.

## Language

- Python 3.11+

## Core libraries

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- umap-learn
- pydantic
- PyYAML

Optional when needed for supported methods:

- pydiffmap or equivalent for Diffusion Maps;
- GPyTorch / GPy / another stable package for GPLVM if practical.

## LLM interface

Implement a provider-agnostic interface.

Do not tightly couple the architecture to one vendor.

Example:

```python
class LLMClient:
    def generate_structured(self, prompt: str, schema):
        ...
```

The concrete provider may be configured separately.

## Agent framework

Do **not** require LangChain, LangGraph, AutoGen, CrewAI, or similar frameworks for V1.

Prefer explicit Python orchestration.

A framework may be added later if it provides a clear benefit.

---

# 6. Repository Structure

Use the following target structure:

```text
DeEntropy/
|
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
|
├── agent/
│   ├── __init__.py
│   ├── state.py
│   ├── schemas.py
│   ├── planner.py
│   ├── reflector.py
│   ├── prompts.py
│   └── llm_client.py
|
├── tools/
│   ├── __init__.py
│   ├── data_inspector.py
│   ├── preprocessing.py
│   ├── dimension_reduction.py
│   ├── evaluation.py
│   └── visualization.py
|
├── reporting/
│   ├── __init__.py
│   └── report_generator.py
|
├── data/
│   ├── dataset_1/
│   └── dataset_2/
|
├── outputs/
│   ├── dataset_1/
│   └── dataset_2/
|
├── logs/
|
├── tests/
│   ├── test_inspector.py
│   ├── test_preprocessing.py
│   ├── test_dimension_reduction.py
│   ├── test_evaluation.py
│   ├── test_state.py
│   └── test_end_to_end.py
|
└── docs/
    └── SYSTEM_DESIGN.md
```

---

# 7. Core Data Flow

The main workflow should be implemented explicitly.

Conceptually:

```python
def run_agent(dataset_path: str, config_path: str = "config.yaml"):
    config = load_config(config_path)

    dataset = load_dataset(dataset_path)

    state = initialize_state(
        dataset_path=dataset_path,
        config=config,
    )

    profile = inspect_dataset(dataset)
    state.dataset_profile = profile

    while not state.finished:
        plan = planner.create_plan(state)

        execution_results = execute_plan(
            dataset=dataset,
            state=state,
            plan=plan,
        )

        evaluation_results = evaluate_results(
            state=state,
            execution_results=execution_results,
        )

        reflection = reflector.reflect(
            state=state,
            evaluation_results=evaluation_results,
        )

        state.update(...)

        if reflection.decision == "stop":
            state.finished = True

        if state.iteration >= state.max_iterations:
            state.finished = True

    generate_report(state)
```

This is conceptual code only. Exact implementation may differ, but the control flow must remain explicit and inspectable.

---

# 8. Agent State

Create an explicit state object in:

```text
agent/state.py
```

Use a dataclass or Pydantic model.

Recommended structure:

```python
class AgentState:
    dataset_path: str

    dataset_profile: DatasetProfile | None

    preprocessing_history: list
    analysis_history: list
    evaluation_history: list
    reflection_history: list

    current_plan: AnalysisPlan | None

    best_embedding_id: str | None

    iteration: int
    max_iterations: int
    finished: bool

    warnings: list[str]
    errors: list[str]
```

The state should provide a single source of truth for what DeEntropy knows and has done.

Do not hide important reasoning or analysis history inside local variables.

---

# 9. Structured Schemas

Create:

```text
agent/schemas.py
```

The LLM must return structured outputs.

Do not parse important decisions from arbitrary free-form prose.

Use Pydantic models or equivalent.

## DatasetProfile

Example:

```python
class DatasetProfile(BaseModel):
    n_samples: int
    n_features: int

    numeric_features: list[str]
    categorical_features: list[str]
    boolean_features: list[str]

    missing_fraction: float
    sparsity: float
    p_to_n_ratio: float

    constant_features: list[str]
    near_constant_features: list[str]

    has_labels: bool
    label_column: str | None

    warnings: list[str]
```

---

## PreprocessingAction

Example:

```python
class PreprocessingAction(BaseModel):
    action: str
    columns: list[str] | None = None
    parameters: dict = {}
    rationale: str
```

Allowed actions should be explicitly validated.

---

## DRMethodPlan

Example:

```python
class DRMethodPlan(BaseModel):
    method: str
    n_components: int = 2
    parameters: dict = {}
    rationale: str
```

---

## AnalysisPlan

Example:

```python
class AnalysisPlan(BaseModel):
    preprocessing: list[PreprocessingAction]
    methods: list[DRMethodPlan]
    evaluation_metrics: list[str]
    analysis_goal: str
    rationale: str
```

---

## ReflectionDecision

Example:

```python
class ReflectionDecision(BaseModel):
    decision: Literal["continue", "stop"]
    reason: str
    suggested_changes: list[str] = []
    preferred_embedding_id: str | None = None
```

---

# 10. Data Loading

For V1, support at minimum:

- CSV.

Optional if easy:

- TSV;
- XLSX.

The system should not require the user to manually identify numeric columns.

The loader should return a pandas DataFrame.

Do not automatically use a label column unless one is explicitly provided through configuration or can be identified through a clearly defined rule.

---

# 11. Data Inspector

Implement:

```text
tools/data_inspector.py
```

Main function:

```python
def inspect_dataset(df: pd.DataFrame, config: dict) -> DatasetProfile:
    ...
```

The inspector should calculate, where applicable:

- sample size \(n\);
- number of features \(p\);
- \(p/n\) ratio;
- numeric feature names;
- categorical feature names;
- boolean feature names;
- missing-value fraction;
- missingness by column;
- sparsity;
- constant features;
- near-constant features;
- duplicate rows;
- basic numeric summaries;
- potential scaling issues;
- high-cardinality categorical features;
- warnings about extremely small \(n\);
- warnings about extremely large \(p\).

The inspector should use deterministic calculations.

It should not call the LLM.

---

# 12. Preprocessing Tools

Implement:

```text
tools/preprocessing.py
```

Expose a controlled interface.

Recommended operations:

```python
remove_constant_features(...)
impute_numeric(...)
impute_categorical(...)
standardize(...)
normalize(...)
log_transform(...)
encode_categorical(...)
select_high_variance_features(...)
```

Also create:

```python
def apply_preprocessing_plan(
    df: pd.DataFrame,
    actions: list[PreprocessingAction],
    random_seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    ...
```

Each action must return metadata describing what changed.

Example:

```json
{
  "action": "standardize",
  "columns": ["x1", "x2", "x3"],
  "status": "success"
}
```

Unknown preprocessing actions should fail safely with a meaningful error.

---

# 13. Dimension-Reduction Tools

Implement:

```text
tools/dimension_reduction.py
```

V1 should provide a common interface:

```python
def run_dimension_reduction(
    X,
    method: str,
    n_components: int,
    parameters: dict,
    random_seed: int,
) -> EmbeddingResult:
    ...
```

## Target supported methods

Implement as many course methods as are practical with stable Python libraries.

Priority order:

1. PCA
2. Sparse PCA or Kernel PCA as the PCA variant
3. MDS
4. Isomap
5. LLE
6. Laplacian Eigenmaps / Spectral Embedding
7. t-SNE
8. UMAP
9. Diffusion Maps
10. GPLVM

Do not block completion of V1 if GPLVM or Diffusion Maps require disproportionate complexity.

If a method is unavailable, the system must clearly report that it is unavailable rather than silently substituting another method.

---

# 14. Embedding Result

Define a structured result.

Example:

```python
class EmbeddingResult(BaseModel):
    embedding_id: str
    method: str
    parameters: dict
    n_components: int

    embedding: Any

    runtime_seconds: float

    success: bool
    error_message: str | None

    metadata: dict
```

The actual embedding matrix may be stored separately if necessary to avoid serialization issues.

---

# 15. Evaluation Tools

Implement:

```text
tools/evaluation.py
```

Use multiple metrics where appropriate.

Recommended metrics:

## Local structure

- trustworthiness;
- k-nearest-neighbor preservation.

## Global structure

- pairwise-distance correlation on a sample when necessary;
- Spearman correlation between original-space and embedding-space distances.

## Reconstruction

Use reconstruction error only for methods for which reconstruction is meaningful.

## Stability

For stochastic methods, optionally rerun with different seeds and compare structure.

Do not treat one universal metric as appropriate for every method.

Create:

```python
def evaluate_embedding(
    X_original,
    embedding_result,
    labels=None,
    config=None,
) -> EvaluationResult:
    ...
```

Recommended structure:

```python
class EvaluationResult(BaseModel):
    embedding_id: str
    metrics: dict[str, float | None]
    warnings: list[str]
    notes: list[str]
```

---

# 16. Visualization

Implement:

```text
tools/visualization.py
```

At minimum generate:

- 2D scatter plot of the embedding;
- plots colored by label if a valid label is available;
- PCA explained-variance plot when PCA is used.

Use matplotlib.

Save plots to the dataset-specific output directory.

Example:

```text
outputs/dataset_1/plots/
    pca_embedding.png
    pca_variance.png
    umap_embedding.png
```

Do not require interactive visualization for V1.

---

# 17. LLM Planner

Implement:

```text
agent/planner.py
```

Main responsibility:

\[
\boxed{\text{What should DeEntropy do next?}}
\]

Inputs should include a compact structured summary of:

- dataset profile;
- preprocessing already performed;
- methods already attempted;
- previous evaluation results;
- current iteration;
- available tools;
- remaining analysis budget.

The Planner must produce a validated `AnalysisPlan`.

The Planner should be instructed to:

- avoid blindly running every algorithm;
- justify preprocessing choices;
- justify method selection;
- avoid repeating failed experiments without reason;
- consider computational feasibility;
- choose a small number of informative candidate methods;
- prefer interpretable baselines such as PCA when appropriate;
- select evaluation metrics appropriate to the selected methods;
- stay within the iteration/experiment budget.

---

# 18. Planner Prompt

Implement prompts in:

```text
agent/prompts.py
```

The planner system prompt should contain principles such as:

```text
You are the planning component of DeEntropy, an automated
dimension-reduction and exploratory-data-analysis agent.

Your job is to decide what statistical analysis should be performed
next based on the dataset profile and previous results.

Do not perform calculations yourself.

Do not invent numerical results.

Do not generate arbitrary Python code.

Select only from the provided preprocessing and dimension-reduction tools.

Do not run every available method by default.

Prefer a small set of methods whose assumptions and strengths match
the observed dataset characteristics.

Return only an AnalysisPlan matching the required schema.
```

The exact wording may be improved, but the constraints must remain.

---

# 19. Reflection Component

Implement:

```text
agent/reflector.py
```

The Reflector decides:

\[
\boxed{
\text{continue analysis}
\quad\text{or}\quad
\text{stop}
}
\]

Inputs:

- dataset profile;
- experiment history;
- evaluation metrics;
- failures/warnings;
- iteration count;
- current best candidate.

The Reflector should ask:

- Is there at least one reasonable embedding?
- Are results numerically acceptable?
- Are competing methods meaningfully different?
- Is further experimentation likely to add useful information?
- Did a method fail because of a fixable hyperparameter or preprocessing issue?
- Has the maximum iteration budget been reached?

Output:

```python
ReflectionDecision
```

The Reflector must not directly execute tools.

---

# 20. Stop Conditions

The system must always have deterministic safety stop conditions even if the LLM wants to continue.

Stop when any of the following occurs:

```text
reflection.decision == "stop"
OR
iteration >= max_iterations
OR
experiment_budget exhausted
OR
no valid next action is available
```

Recommended default:

```yaml
max_iterations: 5
```

Also configure a maximum total number of DR experiments.

Example:

```yaml
max_experiments: 10
```

---

# 21. Main Orchestrator

Implement:

```text
main.py
```

`main.py` is the conductor.

It should:

1. parse arguments;
2. load configuration;
3. load the dataset;
4. create output directories;
5. initialize state;
6. inspect data;
7. run the Plan → Execute → Evaluate → Reflect loop;
8. enforce stopping conditions;
9. generate plots;
10. generate the report;
11. save machine-readable state/history;
12. exit cleanly.

Example CLI:

```bash
python main.py \
    --data data/dataset_1/data.csv \
    --name dataset_1 \
    --config config.yaml
```

---

# 22. Configuration

Create:

```text
config.yaml
```

Recommended fields:

```yaml
random_seed: 42

agent:
  max_iterations: 5
  max_experiments: 10

llm:
  provider: configurable
  model: configurable
  temperature: 0

data:
  label_column: null

output:
  root: outputs

evaluation:
  trustworthiness_neighbors: 10
  distance_sample_size: 1000
  stability_runs: 3
```

Do not hard-code API keys.

API credentials should come from environment variables.

---

# 23. Reproducibility

All stochastic methods must use a controlled random seed whenever the underlying library supports it.

The final report should record:

- random seed;
- package versions if practical;
- preprocessing steps;
- selected methods;
- hyperparameters;
- evaluation metrics.

Save a machine-readable history such as:

```text
outputs/dataset_1/analysis_history.json
```

---

# 24. Logging

Create useful logs.

At minimum record:

```text
timestamp
iteration
planner decision
preprocessing action
DR method
hyperparameters
runtime
evaluation result
reflection decision
warning/error
```

Logs should be saved to:

```text
logs/
```

or inside the dataset-specific output directory.

Do not log API keys or secrets.

---

# 25. Error Handling

The system should not crash because one DR method fails.

Example:

```text
Isomap failed because the neighborhood graph was disconnected.
```

The failure should be captured in the state and returned to the Reflector.

Then the agent may decide to:

- alter the neighborhood parameter;
- apply preprocessing;
- try another method;
- stop using that method.

Tool failure should therefore become **evidence for the next decision**, not necessarily a fatal program error.

Fatal errors should be limited to problems such as:

- unreadable input file;
- no usable features;
- invalid configuration;
- unrecoverable LLM initialization failure.

---

# 26. Report Generator

Implement:

```text
reporting/report_generator.py
```

The generated report should be Markdown for V1.

Expected path:

```text
outputs/dataset_1/generated_report_1.md
```

and:

```text
outputs/dataset_2/generated_report_2.md
```

The report should include:

## Dataset Summary

- \(n\);
- \(p\);
- feature types;
- missingness;
- sparsity;
- other important characteristics.

## Preprocessing

- what was done;
- why it was done.

## Methods Considered

- selected methods;
- why they were selected;
- relevant hyperparameters.

## Results

- evaluation metrics;
- runtime where useful;
- selected visualizations.

## Agent Reflection

- how the agent compared results;
- whether revisions were made;
- why the final embedding(s) were selected.

## Limitations

- uncertainty;
- potential instability;
- method-specific interpretation limitations;
- unresolved data-quality concerns.

## Final Recommendation

Summarize the most useful representation(s) and why.

The report generator may use an LLM for natural-language synthesis, but all numerical facts must come from stored state/results.

The LLM must never invent metrics.

---

# 27. Output Structure

For each dataset:

```text
outputs/dataset_1/
|
├── generated_report_1.md
├── analysis_history.json
├── dataset_profile.json
├── final_state.json
|
├── embeddings/
│   ├── pca.csv
│   └── umap.csv
|
├── metrics/
│   └── evaluation_results.json
|
└── plots/
    ├── pca_embedding.png
    ├── pca_variance.png
    └── umap_embedding.png
```

Equivalent structure should be produced for dataset 2.

---

# 28. Testing Requirements

Create unit tests for deterministic components.

## `test_inspector.py`

Test:

- numeric-only dataset;
- mixed types;
- missing values;
- constant features;
- sparse data.

## `test_preprocessing.py`

Test:

- imputation;
- standardization;
- constant-feature removal;
- invalid action handling.

## `test_dimension_reduction.py`

Test:

- PCA output dimensions;
- UMAP output dimensions when available;
- invalid method handling;
- failure capture.

## `test_evaluation.py`

Test:

- trustworthiness;
- distance metric behavior;
- valid result schema.

## `test_state.py`

Test:

- iteration updates;
- history recording;
- stopping state.

## `test_end_to_end.py`

Use a small synthetic dataset.

The test should run:

\[
\text{load}
\rightarrow
\text{inspect}
\rightarrow
\text{plan}
\rightarrow
\text{execute}
\rightarrow
\text{evaluate}
\rightarrow
\text{reflect}
\rightarrow
\text{report}
\]

without requiring a large external dataset.

For testing, provide a deterministic mock LLM client.

---

# 29. Mock LLM

V1 should support a mock/fake LLM implementation for testing.

Example:

```python
class MockLLMClient:
    def generate_structured(self, prompt, schema):
        ...
```

This allows the full workflow to be tested without:

- API credentials;
- network access;
- nondeterministic LLM behavior.

This is important for reproducibility and continuous testing.

---

# 30. Acceptance Criteria for V1

V1 is complete when all of the following are true.

### Input

- A new CSV dataset can be supplied from the command line.

### Inspection

- DeEntropy automatically generates a dataset profile.

### Planning

- The Planner produces a valid structured analysis plan.

### Preprocessing

- The selected preprocessing operations are executed by controlled tools.

### Dimension reduction

- At least several major course methods are implemented.
- PCA and UMAP should work end-to-end at minimum.
- Unsupported methods fail clearly rather than silently.

### Evaluation

- At least local and global embedding-quality measures are produced.

### Reflection

- The system can decide whether to stop or perform another iteration.

### Safety

- deterministic maximum iteration/experiment limits exist.

### Visualization

- embedding plots are generated automatically.

### Reporting

- a final Markdown report is generated automatically.

### Reproducibility

- decisions, parameters, metrics, and state history are saved.

### Testing

- deterministic components have unit tests;
- an end-to-end test exists using a mock LLM.

---

# 31. Recommended Build Order

Implement in this order:

```text
Phase 1
-------
1. repository scaffold
2. schemas.py
3. state.py
4. data_inspector.py
5. preprocessing.py

Phase 2
-------
6. dimension_reduction.py
7. evaluation.py
8. visualization.py

Phase 3
-------
9. llm_client.py
10. prompts.py
11. planner.py
12. reflector.py

Phase 4
-------
13. main.py
14. report_generator.py
15. logging
16. serialization

Phase 5
-------
17. unit tests
18. mock LLM
19. end-to-end test
20. README
```

Do not begin future-version features before V1 passes the end-to-end test.

---

# 32. Future Roadmap — Do Not Implement Yet

## V2 — Bayesian Optimization

Add:

```text
bayesian/
    __init__.py
    bayesian_optimization.py
```

Purpose:

\[
\boxed{
\text{Bayesian optimization of DR hyperparameters}
}
\]

Potential targets:

- UMAP `n_neighbors`;
- UMAP `min_dist`;
- t-SNE perplexity;
- Isomap neighborhood size;
- LLE neighborhood size.

The objective may combine multiple evaluation metrics.

---

## V3 — RAG

Add:

```text
rag/
    __init__.py
    retriever.py
    knowledge/
        pca.md
        manifold_methods.md
        visualization_methods.md
```

Purpose:

\[
\boxed{
\text{Retrieve methodological knowledge}
\rightarrow
\text{ground Planner decisions}
}
\]

RAG should provide knowledge, not perform numerical analysis.

---

## V4 — Bayesian Sequential Method Selection

Potential model:

\[
P(M\mid X,E)
\]

where:

- \(M\) = candidate dimension-reduction method;
- \(X\) = dataset characteristics;
- \(E\) = evaluation evidence.

The agent could update beliefs after experiments and decide what method to test next.

Do not implement this until a statistically defensible prior and likelihood/update mechanism is defined.

---

## V5 — Multi-Agent Extension

Potential architecture:

```text
Planner Agent
      |
      v
Statistical Tools
      |
      v
Critic Agent
      |
      v
Reporter Agent
```

Only introduce multiple agents if experiments demonstrate a meaningful benefit over the simpler V1 architecture.

---

# 33. Final Architectural Principle

DeEntropy should eventually follow:

\[
\boxed{
\begin{aligned}
\text{Python/statistical rules}
&\rightarrow \text{reliable computation},\\
\text{LLM}
&\rightarrow \text{higher-level reasoning},\\
\text{RAG}
&\rightarrow \text{methodological knowledge},\\
\text{Bayesian methods}
&\rightarrow \text{uncertainty and sequential decisions},\\
\text{Agent loop}
&\rightarrow \text{autonomous adaptation}.
\end{aligned}
}
\]

But **V1 implements only the first, second, and agent-loop components**.

The most important requirement is:

\[
\boxed{
\text{Build a working, explainable, reproducible autonomous analysis pipeline first.}
}
\]

Do not optimize for architectural complexity.

Do not add features merely because they are associated with modern agent systems.

Every component should have a clear statistical or software-engineering purpose.

---

# 34. Initial Codex Instruction

After placing this file in the repository, the recommended initial instruction to Codex is:

```text
Read DeEntropy_CODEX_SPEC.md carefully.

Implement Version 1 of DeEntropy according to this specification.

Do not implement RAG, multi-agent architecture, Bayesian optimization,
Bayesian method selection, reinforcement learning, or other future
extensions.

Work incrementally in the build order defined in the specification.

Keep the architecture simple and explicit. Use deterministic Python
functions for statistical computation and structured LLM outputs for
planning and reflection.

Run the relevant tests after each major component. Do not silently
change interfaces defined in the specification; if an interface must
change for a technical reason, document the reason clearly.

The first milestone is a complete end-to-end pipeline that can process
a small CSV dataset using a mock LLM, produce an embedding, evaluate it,
reflect on the result, and generate a Markdown report.
```
