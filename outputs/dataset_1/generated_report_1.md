# DeEntropy Analysis Report: dataset_1

## Dataset Inspection
- Samples: 801
- Features: 20531
- Numeric features: 20531
- Categorical features: 0
- Boolean features: 0
- Missing-value fraction: 0
- Sparsity: 0.142176
- p/n ratio: 25.6317
- Storage: dense
- Nonzero entries: not computed
- Density: not computed
- Label column: Class

Important data characteristics:
- Dataset has many features: p=20531.
- Feature count exceeds sample count; p/n is greater than 1.
- Constant features detected: gene_5, gene_23, gene_4370, gene_4808, gene_4809, gene_4814, gene_4816, gene_4817, gene_4831, gene_5288, gene_7661, gene_7662, gene_7663, gene_7664, gene_7665, gene_8121, gene_9304, gene_9306, gene_9314, gene_9316, gene_9320, gene_9452, gene_10121, gene_11958, gene_13991, gene_14158, gene_14159, gene_14161, gene_15138, gene_15140, gene_15141, gene_15446, gene_16566, gene_16568, gene_16569, gene_16571, gene_16575, gene_16578, gene_16579, gene_16604, gene_16634, gene_16637, gene_16677, gene_16697, gene_16698, gene_16699, gene_16700, gene_16701, gene_16702, gene_16704, gene_16705, gene_16706, gene_16707, gene_16708, gene_16709, gene_16710, gene_16711, gene_16712, gene_16713, gene_16714, gene_16715, gene_16716, gene_16717, gene_16718, gene_16719, gene_16720, gene_16721, gene_16722, gene_16723, gene_16724, gene_16725, gene_16726, gene_16727, gene_16728, gene_16729, gene_16730, gene_16731, gene_16732, gene_16733, gene_16734, gene_16735, gene_16736, gene_16737, gene_16738, gene_16739, gene_16740, gene_16741, gene_16742, gene_16743, gene_16744, gene_16745, gene_16746, gene_16748, gene_16749, gene_16750, gene_16751, gene_16752, gene_16753, gene_16754, gene_16756, gene_16757, gene_16758, gene_16759, gene_16760, gene_16761, gene_16762, gene_16763, gene_16764, gene_16765, gene_16766, gene_16767, gene_16768, gene_16769, gene_16770, gene_16771, gene_16772, gene_16774, gene_16775, gene_16776, gene_16777, gene_16778, gene_16779, gene_16780, gene_16781, gene_16782, gene_16783, gene_16785, gene_16787, gene_16788, gene_16789, gene_16790, gene_16791, gene_16792, gene_16794, gene_16795, gene_16796, gene_16798, gene_16799, gene_16800, gene_16801, gene_16802, gene_16803, gene_16804, gene_16805, gene_16806, gene_16807, gene_16808, gene_16809, gene_16810, gene_16811, gene_16812, gene_16813, gene_16816, gene_16818, gene_16819, gene_16820, gene_16821, gene_16822, gene_16823, gene_16824, gene_16826, gene_16827, gene_16830, gene_16831, gene_16832, gene_16833, gene_16834, gene_16835, gene_16836, gene_16837, gene_16838, gene_16839, gene_16840, gene_16841, gene_16842, gene_16843, gene_16844, gene_16845, gene_16846, gene_16847, gene_16848, gene_16849, gene_16850, gene_16851, gene_16852, gene_16853, gene_16854, gene_16855, gene_16856, gene_16857, gene_16858, gene_16859, gene_16860, gene_16861, gene_16862, gene_16863, gene_16864, gene_16865, gene_16866, gene_16867, gene_16868, gene_16869, gene_16870, gene_16871, gene_16872, gene_16873, gene_16874, gene_16875, gene_16876, gene_16877, gene_16878, gene_16879, gene_16880, gene_16881, gene_16882, gene_16883, gene_16884, gene_16885, gene_16886, gene_16888, gene_16889, gene_16890, gene_16891, gene_16892, gene_16893, gene_16894, gene_16895, gene_16896, gene_16897, gene_16898, gene_16899, gene_16900, gene_16901, gene_16902, gene_16903, gene_16904, gene_16905, gene_16906, gene_16907, gene_16908, gene_16909, gene_16910, gene_16911, gene_16914, gene_16915, gene_16916, gene_16917, gene_16918, gene_16920, gene_16921, gene_16922, gene_16924, gene_16925, gene_16926, gene_18829, gene_18902, gene_18903, gene_18908, gene_18909, gene_18910, gene_18911, gene_18914, gene_18915, gene_19450, gene_19451, gene_19452, gene_19671.
- Near-constant features detected: gene_9, gene_15, gene_16, gene_1624, gene_1749, gene_1765, gene_1844, gene_2789, gene_2852, gene_3527, gene_4318, gene_4333, gene_4371, gene_4372, gene_4374, gene_4375, gene_4376, gene_4639, gene_4655, gene_4807, gene_4810, gene_4811, gene_4815, gene_4818, gene_4819, gene_4822, gene_4823, gene_4824, gene_4828, gene_4829, gene_4834, gene_4835, gene_6051, gene_6061, gene_6075, gene_6191, gene_6508, gene_6697, gene_6804, gene_7192, gene_7262, gene_7473, gene_7761, gene_7975, gene_9290, gene_9295, gene_9296, gene_9299, gene_9301, gene_9305, gene_9307, gene_9308, gene_9309, gene_9313, gene_9315, gene_9318, gene_9321, gene_9323, gene_9335, gene_9350, gene_9351, gene_9358, gene_9441, gene_9446, gene_9448, gene_9453, gene_9456, gene_9470, gene_9523, gene_9744, gene_9777, gene_10139, gene_10154, gene_11130, gene_11223, gene_11230, gene_12058, gene_12360, gene_12363, gene_12372, gene_12373, gene_12383, gene_12384, gene_12387, gene_12388, gene_12392, gene_12401, gene_12403, gene_12439, gene_12440, gene_12478, gene_12479, gene_12489, gene_12490, gene_12504, gene_12510, gene_12513, gene_12519, gene_12522, gene_12533, gene_12539, gene_12542, gene_12545, gene_12553, gene_12556, gene_12559, gene_12613, gene_12614, gene_12618, gene_12619, gene_12622, gene_12627, gene_12628, gene_12629, gene_12630, gene_12632, gene_12633, gene_12634, gene_12635, gene_12638, gene_12639, gene_12642, gene_12643, gene_12644, gene_12645, gene_12646, gene_12648, gene_12649, gene_12651, gene_12654, gene_12657, gene_12664, gene_12667, gene_12668, gene_12670, gene_12672, gene_12678, gene_12682, gene_12689, gene_12690, gene_12691, gene_12702, gene_12709, gene_12716, gene_12717, gene_12718, gene_12719, gene_12720, gene_12721, gene_12726, gene_12729, gene_12730, gene_12732, gene_12825, gene_13466, gene_13520, gene_13777, gene_13860, gene_13985, gene_14094, gene_14160, gene_14448, gene_14550, gene_14755, gene_14756, gene_14757, gene_14758, gene_14759, gene_14762, gene_15554, gene_15563, gene_15564, gene_16567, gene_16572, gene_16573, gene_16574, gene_16576, gene_16577, gene_16580, gene_16603, gene_16606, gene_16616, gene_16630, gene_16636, gene_16661, gene_16674, gene_16676, gene_16688, gene_16690, gene_16784, gene_16829, gene_16887, gene_16913, gene_17074, gene_17496, gene_17595, gene_17674, gene_17715, gene_17899, gene_18273, gene_18686, gene_18814, gene_18816, gene_18817, gene_18904, gene_18913, gene_18916, gene_18918, gene_18924, gene_18992.
- Numeric features appear to be on substantially different scales.
- The high feature-to-sample ratio can make pairwise-distance and manifold methods expensive; consider feature selection.

## Preprocessing

- Pipeline `preprocessing_bc5a3a13b6d12db6`: `raw_X` -> `preprocessed_X`
  - remove_constant_features: 267 feature(s) [gene_5, gene_23, gene_4370, ...]; status=success
    Rationale: Remove constant features because they cannot contribute to an embedding.
  - select_high_variance_features: 4056 feature(s) [gene_9176, gene_9175, gene_15898, ...]; status=success
    Parameters: `{'mode': 'top_k', 'threshold': 0.0, 'top_k': 4056, 'removed': '<16208 values>'}`
    Rationale: Select variance-ranked features from the transformed representation to reduce high-dimensional computation.
  - standardize: 4056 feature(s) [gene_9176, gene_9175, gene_15898, ...]; status=success
    Rationale: Put usable numeric features on a comparable scale before distance-based methods.

## Methods and Rationales

- Iteration 0: pca (pca_2313984036); success=True
  Rationale: Use PCA to create a size-derived intermediate representation compatible with downstream analysis.
  Parameters: `{'random_state': 42, 'svd_solver': 'auto'}`
- Iteration 0: isomap (isomap_12a2f0fb81); success=True
  Rationale: Apply Isomap to the dense intermediate because the sample count permits a geodesic-neighborhood comparison.
  Parameters: `{'n_neighbors': 15}`
- Iteration 1: spectral_embedding (spectral_embedding_820542773d); success=True
  Rationale: Reuse the completed pca artifact and apply Spectral Embedding / Laplacian Eigenmaps because the sample count permits graph spectral analysis.
  Parameters: `{'n_neighbors': 15, 'random_state': 42, 'affinity': 'nearest_neighbors'}`

## Intrinsic Embedding Evaluation

- Embedding: isomap_12a2f0fb81
  - trustworthiness: 0.942195
  - knn_preservation: 0.242072
  - distance_pearson_correlation: 0.752909
  - distance_spearman_correlation: 0.754262
  - Note: External validation uses labels only after embedding creation; labels were not used to fit or tune the embedding.
- Embedding: spectral_embedding_820542773d
  - trustworthiness: 0.956825
  - knn_preservation: 0.299126
  - distance_pearson_correlation: 0.680257
  - distance_spearman_correlation: 0.724437
  - Note: External validation uses labels only after embedding creation; labels were not used to fit or tune the embedding.

## External Label-Based Validation

Labels were used only for post-hoc validation and visualization, not as dimension-reduction features, preprocessing targets, or tuning objectives.

- Embedding: isomap_12a2f0fb81
  - silhouette_score: 0.757532
  - neighborhood_label_agreement: 0.989139
- Embedding: spectral_embedding_820542773d
  - silhouette_score: 0.882656
  - neighborhood_label_agreement: 0.995381

## Agent Reflection

- Reflection 1: continue - Current results do not yet show clearly acceptable local-structure preservation.
  Preferred embedding: isomap_12a2f0fb81
  Suggested changes: try method umap, adjust n_neighbors
- Reflection 2: stop - Experiment budget is exhausted.
  Preferred embedding: spectral_embedding_820542773d

## Limitations

- Automated method selection is limited to the configured V1 tool set.
- Metrics summarize structural preservation but do not prove biological or domain significance.
- Label alignment can describe correspondence with known classes but cannot establish mechanisms or causal explanations.
- Stochastic methods may vary despite controlled random seeds.

## Final Recommendation

Use `spectral_embedding_820542773d` for local-neighborhood exploration.

- For local-neighborhood exploration: `spectral_embedding_820542773d`. Compared kNN preservation first, then trustworthiness; both assess local-neighborhood preservation without combining them into one score. Recorded method context: External validation uses labels only after embedding creation; labels were not used to fit or tune the embedding.
- For broader pairwise-structure reference: `isomap_12a2f0fb81`. Compared Spearman distance correlation first, then Pearson distance correlation; these summarize broader pairwise-distance preservation. Recorded method context: External validation uses labels only after embedding creation; labels were not used to fit or tune the embedding.
- For post-hoc alignment with known labels: `spectral_embedding_820542773d`. Compared label silhouette first, then neighborhood label agreement. Labels were used only for external validation. Recorded method context: External validation uses labels only after embedding creation; labels were not used to fit or tune the embedding.

Tradeoffs:
- `spectral_embedding_820542773d` is preferred for local-neighborhood exploration, while `isomap_12a2f0fb81` is complementary for broader pairwise-structure preservation.

## Reproducibility

- Random seed: 42
- Max iterations: 5
- Max experiments: 2
