# DeEntropy

DeEntropy is an automated dimension-reduction and exploratory-data-analysis
agent. It profiles a dataset, proposes a constrained analysis plan, executes
deterministic statistical tools, evaluates terminal embeddings, reflects on
the evidence, and writes machine-readable outputs and a Markdown report.

The governing rule is simple: **the LLM proposes; deterministic code validates
and executes**. An LLM cannot run code, alter data outside registered
preprocessing actions, bypass budgets, or introduce an unsupported method.

## Architecture

```text
Dataset -> Inspect -> Plan -> Preprocess -> Execute graph
        -> Evaluate terminal embeddings -> Reflect -> Replan/Stop -> Report
```

- `tools/analysis_dataset.py`: shared dense/sparse dataset abstraction and CSV
  or Matrix Market loading.
- `tools/data_inspector.py`: deterministic dataset profiling.
- `agent/method_registry.py`: method capabilities, feasibility rules, and
  parameter validation.
- `agent/planner.py`: structured planning plus deterministic guardrails.
- `tools/preprocessing.py`: registered dense and sparse-safe transformations.
- `tools/dimension_reduction.py`: numerical method execution.
- `tools/evaluation.py` and `tools/visualization.py`: metrics and plots.
- `agent/reflector.py`: structured continuation decisions and final synthesis.
- `reporting/report_generator.py`: Markdown report generation.
- `main.py`: CLI, artifact graph execution, provenance, budgets, and persistence.

Plans are acyclic artifact graphs. A terminal experiment may consume
`preprocessed_X` directly or a reusable intermediate representation:

```text
raw_X -> preprocessing -> preprocessed_X -> PCA_k -> terminal embedding
```

Experiment budgets count terminal strategies. Intermediate transformations
count toward a separate analysis-step safety limit and may be shared by more
than one terminal experiment.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

Dependencies include NumPy, pandas, SciPy, scikit-learn, matplotlib,
umap-learn, Pydantic, PyYAML, the OpenAI Python SDK, and pytest.

## Input Formats

### CSV

CSV input supports optional IDs and labels in the feature file or a separate
labels file:

```yaml
data:
  labels_path: path/to/labels.csv
  id_column: sample_id
  label_column: Class
```

Separate labels are joined one-to-one by ID while preserving feature-matrix
row order. IDs are retained as provenance. ID and label columns are excluded
from preprocessing, planning statistics, feature selection, and embedding
fitting. Labels are used only for plot coloring and post-hoc validation.

### Matrix Market

Generic Matrix Market bundles remain sparse and are standardized internally to
samples by features:

```yaml
data:
  format: matrix_market
  orientation: features_by_samples
  sample_metadata_path: path/to/samples.tsv
  feature_metadata_path: path/to/features.tsv
  sample_id_column: 0
  feature_id_column: 0
  feature_name_column: 1
  metadata_delimiter: "\t"
  metadata_has_header: false
```

Stable sample and feature IDs must be unique. Duplicate display names are
allowed. The loader validates orientation, dimensions, and metadata alignment
without densifying the complete sparse matrix.

## Configuration

Important fields are:

```yaml
random_seed: 42

agent:
  max_iterations: 5
  max_experiments: 2
  max_analysis_steps: 20

llm:
  provider: mock
  model: mock
  temperature: 0

output:
  root: outputs
```

`max_experiments` limits terminal strategies; `max_analysis_steps` limits all
executed graph steps. Dataset-specific configurations are examples of the same
general architecture, not hard-coded workflows.

## Running DeEntropy

Deterministic mock mode requires no credentials or network access:

```bash
python main.py \
  --data data/dataset_1/data.csv \
  --name dataset_1 \
  --config config_dataset_1.yaml \
  --llm-provider mock
```

For Matrix Market input, pass the `.mtx` file as `--data` and provide metadata
paths and orientation in the selected YAML configuration.

### OpenAI Provider

The real provider uses the OpenAI Responses API with Pydantic structured
outputs. Configure a concrete model and keep credentials in the environment:

```yaml
llm:
  provider: openai
  model: YOUR_SUPPORTED_MODEL
  temperature: 0
  api_key_env: OPENAI_API_KEY
```

```bash
export OPENAI_API_KEY="your-api-key"
python main.py --data path/to/data.csv --name my_run --config my_config.yaml
```

No key is read in mock mode. Provider output is schema-validated and then sent
through the same registry, feasibility, graph, parameter, and budget checks as
mock output. Invalid output fails before execution. Unit tests mock the SDK and
never make paid API requests.

## Preprocessing

Registered operations include:

- numeric and categorical imputation;
- constant-feature removal;
- dense standardization and min-max normalization;
- categorical encoding;
- dense or sparse sample-total normalization for nonnegative count-like data;
- sparse-safe `log1p` where compatible;
- variance-threshold or deterministic top-k feature selection.

Operations declare dense/sparse compatibility. Unsupported sparse operations
fail clearly rather than silently densifying a large matrix.

## Dimension Reduction

Executable methods:

- PCA;
- Truncated SVD, including sparse input;
- Sparse PCA;
- Kernel PCA;
- MDS;
- Isomap;
- LLE;
- Spectral Embedding / Laplacian Eigenmaps;
- t-SNE;
- UMAP.

PCA, Truncated SVD, Sparse PCA, and Kernel PCA can be planned as reusable
intermediate or terminal representations where supported. Registry metadata
normally treats MDS, Isomap, LLE, Spectral Embedding, t-SNE, and UMAP as
terminal embeddings. Diffusion Maps and GPLVM are known to the registry but
explicitly unsupported and non-executable.

## Evaluation And Reflection

Intrinsic metrics are trustworthiness, k-nearest-neighbor preservation, and
Pearson and Spearman distance correlation.

For composed workflows, provenance distinguishes fidelity to the immediate
parent representation from fidelity to `preprocessed_X`. Sparse-root fidelity
is omitted when it would require densifying the sparse root.

When labels exist, silhouette score and neighborhood label agreement are
computed after embedding creation. They are external validation signals, never
fitting or tuning objectives.

The reflector receives completed terminal experiments, actual evaluations,
failures, artifacts, and remaining budgets. It may request a distinct feasible
experiment or stop. Final recommendations are purpose-specific: local
neighborhood preservation, broader pairwise structure, and optional post-hoc
label alignment are not averaged into one score.

## Outputs

Each run writes beneath `outputs/<name>/`:

```text
dataset_profile.json
analysis_history.json
final_state.json
generated_report_<dataset-number>.md
run.log
embeddings/*.csv
metrics/evaluation_results.json
plots/*.png
```

Embedding CSVs include `sample_id` followed by component columns. State,
history, metrics, artifacts, and reports retain method and reference-space
provenance. Non-numbered custom run names use `generated_report.md`.

## Reproducibility And Tests

Seeds are propagated to stochastic methods where supported. Configurations,
parameters, analysis history, artifact provenance, and evaluation results are
serialized with each run. The deterministic mock provider is used for offline
reproducibility and end-to-end testing.

```bash
python -m pytest -q
python -m compileall -q agent tools reporting main.py
```

## Limitations

- Mock planning uses transparent deterministic heuristics; real LLM planning
  can still be uncertain, although deterministic guardrails remain authoritative.
- Experiment budgets deliberately limit method coverage.
- Diffusion Maps and GPLVM are not implemented.
- Sparse-root fidelity is unavailable when evaluation would require full
  densification.
- Preprocessing is general-purpose rather than domain-specialized; there is no
  clustering, cell annotation, or specialized highly-variable-gene procedure.
- External label validation is unavailable for unlabeled datasets.
- Structural metrics support exploratory comparison but do not establish
  biological validity, mechanisms, or causal conclusions.
- Stochastic library behavior can vary across dependency versions or platforms
  even when seeds are supplied.

RAG, Bayesian decision making, multi-agent coordination, reinforcement
learning, clustering, and generative modeling are outside Project 1 scope.
