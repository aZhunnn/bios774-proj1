import numpy as np

from tools.dimension_reduction import run_dimension_reduction
from tools.visualization import generate_embedding_plots, plot_embedding, plot_pca_explained_variance


def make_matrix(n_samples=24, n_features=4):
    rng = np.random.default_rng(456)
    return rng.normal(size=(n_samples, n_features))


def test_plot_embedding_creates_png(tmp_path):
    result = run_dimension_reduction(make_matrix(), "pca", 2, {}, random_seed=42)

    output_path = plot_embedding(result, tmp_path, labels=["a", "b"] * 12)

    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0


def test_plot_pca_explained_variance_creates_png(tmp_path):
    result = run_dimension_reduction(make_matrix(), "pca", 2, {}, random_seed=42)

    output_path = plot_pca_explained_variance(result, tmp_path)

    assert output_path.exists()
    assert output_path.name == "pca_variance.png"
    assert output_path.stat().st_size > 0


def test_generate_embedding_plots_creates_standard_outputs(tmp_path):
    result = run_dimension_reduction(make_matrix(), "pca", 2, {}, random_seed=42)

    paths = generate_embedding_plots([result], tmp_path)

    assert len(paths) == 2
    assert all(path.exists() for path in paths)

