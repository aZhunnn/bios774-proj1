# DeEntropy Analysis Report: dataset_2

## Dataset Inspection
- Samples: 2700
- Features: 32738
- Numeric features: 32738
- Categorical features: 0
- Boolean features: 0
- Missing-value fraction: 0
- Sparsity: 0.974128
- p/n ratio: 12.1252
- Storage: sparse
- Nonzero entries: 2286884
- Density: 0.0258719
- Label column: None

Important data characteristics:
- Dataset has many features: p=32738.
- Constant features detected: 16104.
- Near-constant sparse features detected: 7781.
- Sparse storage must be preserved; dense conversion may exceed practical memory.
- The high feature-to-sample ratio can make pairwise-distance and manifold methods expensive; consider feature selection.
- Numeric features appear to be on substantially different scales.

## Preprocessing

- Pipeline `preprocessing_1b850d7ec2ee67ac`: `raw_X` -> `preprocessed_X`
  - remove_constant_features: 16104 feature(s) [ENSG00000243485, ENSG00000237613, ENSG00000186092, ...]; status=success
    Rationale: Remove constant features because they cannot contribute to an embedding.
  - normalize_sample_totals: 16634 feature(s) [ENSG00000237683, ENSG00000228463, ENSG00000228327, ...]; status=success
    Parameters: `{'target_total': 1.0}`
    Rationale: Normalize varying nonnegative sample totals before comparing feature magnitudes across samples.
  - log_transform: 16634 feature(s) [ENSG00000237683, ENSG00000228463, ENSG00000228327, ...]; status=success
    Parameters: `{'offset': 1.0}`
    Rationale: Apply sparse-safe log1p after sample-total normalization because the count-like values show scale or sparsity-related skew evidence.
  - select_high_variance_features: 9402 feature(s) [ENSG00000188976, ENSG00000188290, ENSG00000187608, ...]; status=success
    Parameters: `{'mode': 'top_k', 'top_k': 9402, 'threshold': 0.0, 'removed': '<7232 values>'}`
    Rationale: Select variance-ranked features from the transformed representation to reduce high-dimensional computation.

## Methods and Rationales

- Iteration 0: truncated_svd (truncated_svd_70e9be356b); success=True
  Rationale: Use Truncated SVD to create a size-derived intermediate representation compatible with downstream analysis.
  Parameters: `{'random_state': 42, 'algorithm': 'randomized', 'n_iter': 5}`
- Iteration 0: umap (umap_f83633334a); success=True
  Rationale: Apply UMAP to the dense intermediate because graph-based local exploration is preferable to quadratic methods at larger n.
  Parameters: `{'n_neighbors': 15, 'random_state': 42, 'n_jobs': 1, 'min_dist': 0.1}`
- Iteration 1: pca (pca_0265a858c8); success=True
  Rationale: Reuse the completed truncated_svd artifact and apply PCA because an efficient linear baseline remains useful at larger n.
  Parameters: `{'random_state': 42, 'svd_solver': 'auto'}`

## Intrinsic Embedding Evaluation

- Embedding: umap_f83633334a
  - trustworthiness: 0.921947
  - knn_preservation: 0.213111
  - distance_pearson_correlation: 0.735086
  - distance_spearman_correlation: 0.785862
  - Note: umap emphasizes local neighborhoods; global distance correlations should be interpreted cautiously.
- Embedding: pca_0265a858c8
  - trustworthiness: 0.864981
  - knn_preservation: 0.0877037
  - distance_pearson_correlation: 0.908476
  - distance_spearman_correlation: 0.890142
  - Note: PCA-family embeddings have method-specific variance or component metadata when available.

## External Label-Based Validation

No label-based external validation metrics were available.

## Agent Reflection

- Reflection 1: continue - Current results do not yet show clearly acceptable local-structure preservation.
  Preferred embedding: umap_f83633334a
  Suggested changes: try method umap, adjust n_neighbors
- Reflection 2: stop - Experiment budget is exhausted.
  Preferred embedding: umap_f83633334a

## Limitations

- Automated method selection is limited to the configured V1 tool set.
- Metrics summarize structural preservation but do not prove biological or domain significance.
- Label alignment can describe correspondence with known classes but cannot establish mechanisms or causal explanations.
- Stochastic methods may vary despite controlled random seeds.

## Final Recommendation

Use `umap_f83633334a` for local-neighborhood exploration.

- For local-neighborhood exploration: `umap_f83633334a`. Compared kNN preservation first, then trustworthiness; both assess local-neighborhood preservation without combining them into one score. Recorded method context: umap emphasizes local neighborhoods; global distance correlations should be interpreted cautiously.
- For broader pairwise-structure reference: `pca_0265a858c8`. Compared Spearman distance correlation first, then Pearson distance correlation; these summarize broader pairwise-distance preservation. Recorded method context: PCA-family embeddings have method-specific variance or component metadata when available.

Tradeoffs:
- `umap_f83633334a` is preferred for local-neighborhood exploration, while `pca_0265a858c8` is complementary for broader pairwise-structure preservation.

## Reproducibility

- Random seed: 42
- Max iterations: 5
- Max experiments: 2
