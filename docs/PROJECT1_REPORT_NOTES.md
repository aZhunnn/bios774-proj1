# Project 1 Report Notes

This document summarizes the finalized stored results. It is preparation
material for the manual report, not the final four-page report.

## Architecture

DeEntropy follows `Inspect -> Plan -> Preprocess -> Execute -> Evaluate ->
Reflect -> Replan/Stop -> Report`. `AnalysisDataset` preserves dense or sparse
matrices, sample IDs, feature IDs, metadata, and optional labels. The planner
receives a compact `DatasetProfile`, capability registry, feasibility results,
budgets, prior experiments, artifacts, and reflector feedback. It returns a
Pydantic `AnalysisPlan`.

Deterministic code validates preprocessing compatibility and order, executable
methods, hyperparameters, graph references and acyclicity, representation
compatibility, terminal-experiment budget, and total analysis-step limit. The
executor alone performs numerical work. Terminal experiments are evaluated and
compared; intermediate artifacts retain provenance and can be reused without
counting as additional terminal experiments.

The reflector receives actual terminal histories and metrics. Final synthesis
separates local-neighborhood, broader pairwise-structure, and optional external
label-alignment recommendations rather than averaging unlike metrics.

## Dataset 1: Gene Expression Cancer RNA-Seq

Profile: 801 samples by 20,531 numeric features, dense storage, no missing
values, no duplicate rows, `p/n = 25.6317`, 267 constant features, and 210
near-constant features. Five class labels were available separately.

Final preprocessing:

1. Remove 267 constant features: 20,531 to 20,264 features.
2. Select the top 4,056 variance-ranked features.
3. Standardize the selected features.

The reusable intermediate was PCA with 29 components. Terminal experiments
were Isomap (`n_neighbors=15`) and Spectral Embedding / Laplacian Eigenmaps
(`affinity=nearest_neighbors`, `n_neighbors=15`).

### Dataset 1 Metrics

| Embedding | Reference | Trustworthiness | kNN preservation | Pearson distance | Spearman distance |
|---|---|---:|---:|---:|---:|
| Isomap | PCA-29 parent | 0.942195 | 0.242072 | 0.752909 | 0.754262 |
| Isomap | Preprocessed root | 0.928590 | 0.197004 | 0.672473 | 0.662908 |
| Spectral Embedding | PCA-29 parent | 0.956825 | 0.299126 | 0.680257 | 0.724437 |
| Spectral Embedding | Preprocessed root | 0.944090 | 0.239326 | 0.598923 | 0.628195 |

Post-hoc external validation:

| Embedding | Silhouette | Neighborhood label agreement |
|---|---:|---:|
| Isomap | 0.757532 | 0.989139 |
| Spectral Embedding | 0.882656 | 0.995381 |

Final recommendations: Spectral Embedding for local-neighborhood exploration
and post-hoc alignment with known labels; Isomap as the broader
pairwise-structure reference. Labels were excluded from preprocessing,
selection, fitting, and tuning and used only after embedding creation.

## Dataset 2: PBMC3K Matrix

Profile: 2,700 samples by 32,738 features, CSR sparse storage, 2,286,884
nonzero entries, density 0.025872, sparsity 0.974128, `p/n = 12.1252`,
nonnegative integer/count-like values, 16,104 constant features, and 7,781
near-constant features. Sample totals ranged from 548 to 15,844. No class or
cell-type labels were available.

Final preprocessing:

1. Remove constant features: 32,738 to 16,634 features.
2. Normalize each sample total to 1.0.
3. Apply sparse-safe `log1p` with offset 1.0.
4. Select the top 9,402 variance-ranked features.

The sparse matrix was never fully densified. The reusable intermediate was
noncentered Truncated SVD with 52 components. Terminal experiments were UMAP
(`n_neighbors=15`, `min_dist=0.1`) and PCA (`svd_solver=auto`), both applied to
the dense SVD representation.

### Dataset 2 Metrics

| Embedding | Trustworthiness | kNN preservation | Pearson distance | Spearman distance |
|---|---:|---:|---:|---:|
| UMAP | 0.921947 | 0.213111 | 0.735086 | 0.785862 |
| PCA | 0.864981 | 0.087704 | 0.908476 | 0.890142 |

These are parent-fidelity metrics relative to SVD-52. Root fidelity was not
computed because it would require densifying the sparse preprocessed root.
Final recommendations: UMAP for local-neighborhood exploration and PCA as the
broader pairwise-structure reference. External label validation was unavailable.

## Supported Strengths

- One dense and one sparse real-data workflow use the same general architecture.
- Planning is profile-aware and constrained by explicit method capabilities.
- Sparse loading, inspection, preprocessing, and Truncated SVD avoid full-root
  densification.
- Artifact graphs support independent or composed workflows and exact
  intermediate reuse.
- Terminal experiment budgets are distinct from analysis-step limits.
- Intrinsic and external metrics remain separate, with parent/root provenance.
- Sample IDs are preserved in every exported representation.
- Mock mode provides deterministic, network-free end-to-end reproducibility.
- A real OpenAI adapter uses structured outputs while deterministic validation
  remains authoritative.

## Limitations

- Mock decisions are heuristic; real LLM proposals may be uncertain or invalid
  and therefore require deterministic rejection.
- Two-terminal-experiment budgets provide comparison but not exhaustive search.
- Diffusion Maps and GPLVM are registered but not executable.
- Sparse-root fidelity is unavailable when it would require full densification.
- Preprocessing is general rather than domain-specialized; no specialized HVG
  model, clustering, or biological annotation is included.
- Dataset 2 has no labels, so no external validation was possible.
- Metrics measure structural preservation, not biological truth or causality.
- Dataset 1 Spectral Embedding plot title exposes the canonical underscore name;
  it is readable but less polished than its registry display name.

## Reproducibility

- Final runs used random seed 42 and deterministic mock planning.
- Dataset-specific YAML files record input, budgets, inspection, and evaluation
  configuration.
- `dataset_profile.json`, `analysis_history.json`, `final_state.json`, embedding
  CSVs, evaluation JSON, plots, logs, and Markdown reports preserve provenance.
- Dataset 1 exports 801 aligned sample IDs; Dataset 2 exports 2,700.
- The automated test suite covers schemas, guardrails, dense/sparse inputs,
  preprocessing, methods, evaluation, artifact graphs, orchestration, reports,
  and network-free structured-provider behavior.
