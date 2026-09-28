"""Dimension-reduction capabilities, feasibility checks, and parameter guardrails."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterCapability:
    value_type: str
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class MethodCapability:
    canonical_name: str
    display_name: str
    family: str
    supported_roles: tuple[str, ...]
    dense_input: bool
    sparse_input: bool
    scalability: str
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    relevant_characteristics: tuple[str, ...]
    parameters: dict[str, ParameterCapability]
    executable: bool = True
    quadratic_sample_scaling: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parameter(
    value_type: str,
    default: Any = None,
    minimum: float | None = None,
    maximum: float | None = None,
    choices: tuple[Any, ...] = (),
    description: str = "",
) -> ParameterCapability:
    return ParameterCapability(
        value_type=value_type,
        default=default,
        minimum=minimum,
        maximum=maximum,
        choices=choices,
        description=description,
    )


NEIGHBOR_PARAMETER = _parameter(
    "integer", minimum=2, description="Must be smaller than the sample count."
)

METHOD_CAPABILITIES: dict[str, MethodCapability] = {
    "truncated_svd": MethodCapability(
        "truncated_svd", "Truncated SVD", "linear_noncentered", ("terminal", "intermediate"), True, True,
        "Sparse-safe linear reduction that avoids centering the input matrix.",
        ("Accepts sparse matrices without densifying the original input", "Produces a reusable dense representation"),
        ("Components describe uncentered variance and must not be interpreted as ordinary PCA",),
        ("sparse data", "high-dimensional data", "intermediate linear representation"),
        {
            "algorithm": _parameter("string", "randomized", choices=("randomized", "arpack")),
            "n_iter": _parameter("integer", 5, minimum=1),
            "tol": _parameter("number", 0.0, minimum=0.0),
        },
    ),
    "pca": MethodCapability(
        "pca", "PCA", "linear", ("terminal", "intermediate"), True, False,
        "Efficient on ordinary dense matrices; randomized solvers help when dimensions are large.",
        ("Interpretable linear baseline", "Explained-variance diagnostics"),
        ("Captures only linear variance", "Centers dense input"),
        ("numeric data", "correlated features", "high feature-to-sample ratio"),
        {
            "svd_solver": _parameter("string", "auto", choices=("auto", "full", "covariance_eigh", "arpack", "randomized")),
            "whiten": _parameter("boolean", False),
            "tol": _parameter("number", 0.0, minimum=0.0),
            "iterated_power": _parameter("integer_or_auto", "auto", minimum=0),
        },
    ),
    "sparse_pca": MethodCapability(
        "sparse_pca", "Sparse PCA", "linear_sparse_loadings", ("terminal", "intermediate"), True, False,
        "Iterative optimization can be expensive as samples, features, or components grow.",
        ("Sparse, potentially interpretable component loadings",),
        ("Does not accept sparse storage in the current executor", "No explained-variance ratio"),
        ("high-dimensional data", "feature interpretability"),
        {
            "alpha": _parameter("number", 1.0, minimum=0.0),
            "ridge_alpha": _parameter("number", 0.01, minimum=0.0),
            "max_iter": _parameter("integer", 1000, minimum=1),
            "tol": _parameter("number", 1e-8, minimum=0.0),
            "method": _parameter("string", "lars", choices=("lars", "cd")),
        },
    ),
    "kernel_pca": MethodCapability(
        "kernel_pca", "Kernel PCA", "kernel", ("terminal", "intermediate"), True, False,
        "Constructs an n-by-n kernel matrix and therefore has quadratic sample memory.",
        ("Can represent nonlinear structure through a selected kernel",),
        ("Kernel choice and scale strongly affect results", "Quadratic sample memory"),
        ("moderate sample size", "possible nonlinear structure"),
        {
            "kernel": _parameter("string", "rbf", choices=("linear", "poly", "rbf", "sigmoid", "cosine")),
            "gamma": _parameter("number_or_none", None, minimum=0.0),
            "degree": _parameter("integer", 3, minimum=1),
            "coef0": _parameter("number", 1.0),
            "eigen_solver": _parameter("string", "auto", choices=("auto", "dense", "arpack", "randomized")),
        },
        quadratic_sample_scaling=True,
    ),
    "mds": MethodCapability(
        "mds", "MDS", "distance_geometry", ("terminal",), True, False,
        "Computes pairwise dissimilarities and iteratively optimizes stress; quadratic in samples.",
        ("Directly represents pairwise distance geometry", "Provides stress"),
        ("Expensive for large sample counts", "Sensitive to local minima"),
        ("small sample size", "global distance exploration"),
        {
            "metric": _parameter("boolean", True),
            "n_init": _parameter("integer", 1, minimum=1),
            "max_iter": _parameter("integer", 300, minimum=1),
            "eps": _parameter("number", 1e-3, minimum=0.0),
        },
        quadratic_sample_scaling=True,
    ),
    "isomap": MethodCapability(
        "isomap", "Isomap", "manifold", ("terminal",), True, False,
        "Builds a neighbor graph, computes graph shortest paths, and performs an eigendecomposition.",
        ("Preserves estimated geodesic manifold distances",),
        ("Sensitive to neighborhood size and disconnected graphs",),
        ("moderate sample size", "continuous manifold structure"),
        {
            "n_neighbors": NEIGHBOR_PARAMETER,
            "eigen_solver": _parameter("string", "auto", choices=("auto", "arpack", "dense")),
            "path_method": _parameter("string", "auto", choices=("auto", "FW", "D")),
            "metric": _parameter("string", "minkowski"),
        },
    ),
    "lle": MethodCapability(
        "lle", "LLE", "manifold", ("terminal",), True, False,
        "Builds local reconstruction neighborhoods followed by a global eigendecomposition.",
        ("Models locally linear manifold structure",),
        ("Sensitive to noise, graph connectivity, and neighborhood size",),
        ("moderate sample size", "locally linear manifold structure"),
        {
            "n_neighbors": NEIGHBOR_PARAMETER,
            "method": _parameter("string", "standard", choices=("standard", "hessian", "modified", "ltsa")),
            "reg": _parameter("number", 1e-3, minimum=0.0),
            "eigen_solver": _parameter("string", "auto", choices=("auto", "arpack", "dense")),
            "max_iter": _parameter("integer", 100, minimum=1),
        },
    ),
    "spectral_embedding": MethodCapability(
        "spectral_embedding", "Spectral Embedding / Laplacian Eigenmaps", "spectral", ("terminal",), True, False,
        "Builds an affinity graph and computes graph eigenvectors.",
        ("Represents graph and local-neighborhood structure",),
        ("Sensitive to affinity construction and disconnected graphs",),
        ("moderate sample size", "graph-like or manifold structure"),
        {
            "n_neighbors": NEIGHBOR_PARAMETER,
            "affinity": _parameter("string", "nearest_neighbors", choices=("nearest_neighbors", "rbf")),
            "gamma": _parameter("number_or_none", None, minimum=0.0),
            "eigen_solver": _parameter("string_or_none", None, choices=(None, "arpack", "lobpcg", "amg")),
        },
    ),
    "tsne": MethodCapability(
        "tsne", "t-SNE", "local_visualization", ("terminal",), True, False,
        "Barnes-Hut mode scales better than exact t-SNE but remains costly for large datasets.",
        ("Strong local cluster and neighborhood visualization",),
        ("Global distances are not directly interpretable", "No reusable transform in sklearn"),
        ("local visualization", "moderate sample size"),
        {
            "perplexity": _parameter("number", 30.0, minimum=0.0, description="Must be smaller than the sample count."),
            "learning_rate": _parameter("number_or_auto", "auto", minimum=0.0),
            "init": _parameter("string", "pca", choices=("pca", "random")),
            "max_iter": _parameter("integer", 1000, minimum=250),
            "method": _parameter("string", "barnes_hut", choices=("barnes_hut", "exact")),
            "angle": _parameter("number", 0.5, minimum=0.0, maximum=1.0),
        },
    ),
    "umap": MethodCapability(
        "umap", "UMAP", "local_manifold", ("terminal",), True, False,
        "Neighbor-graph construction is usually more scalable than quadratic manifold methods.",
        ("Flexible local-neighborhood visualization", "Supports several distance metrics"),
        ("Global geometry can be distorted", "Stochastic optimization"),
        ("larger sample size", "local manifold exploration"),
        {
            "n_neighbors": NEIGHBOR_PARAMETER,
            "min_dist": _parameter("number", 0.1, minimum=0.0, maximum=1.0),
            "spread": _parameter("number", 1.0, minimum=0.0),
            "metric": _parameter("string", "euclidean"),
            "n_epochs": _parameter("integer_or_none", None, minimum=1),
            "learning_rate": _parameter("number", 1.0, minimum=0.0),
        },
    ),
    "diffusion_maps": MethodCapability(
        "diffusion_maps", "Diffusion Maps", "diffusion", ("terminal", "intermediate"), True, False,
        "No stable implementation is enabled in DeEntropy V1.", (),
        ("Currently unsupported",), ("diffusion geometry",), {}, executable=False,
    ),
    "gplvm": MethodCapability(
        "gplvm", "GPLVM", "probabilistic_latent", ("terminal", "intermediate"), True, False,
        "No probabilistic latent-variable implementation is enabled in DeEntropy V1.", (),
        ("Currently unsupported",), ("probabilistic latent representation",), {}, executable=False,
    ),
}


QUADRATIC_SAMPLE_LIMIT = 2_000


def executable_method_names() -> list[str]:
    return [name for name, capability in METHOD_CAPABILITIES.items() if capability.executable]


def method_capability_context() -> dict[str, dict[str, Any]]:
    return {name: capability.to_dict() for name, capability in METHOD_CAPABILITIES.items()}


def rank_feasible_methods(
    profile: dict[str, Any],
    feasibility: dict[str, dict[str, Any]],
) -> list[tuple[str, int, str]]:
    """Rank feasible methods using profile evidence, not dataset identity."""
    n_samples = int(profile.get("n_samples") or 0)
    n_features = int(profile.get("n_features") or 0)
    ratio = float(profile.get("p_to_n_ratio") or 0)
    scores: dict[str, tuple[int, str]] = {
        "truncated_svd": (15, "a noncentered linear representation"),
        "pca": (55, "an interpretable linear baseline with variance diagnostics"),
        "sparse_pca": (20, "sparse loadings can aid feature-level interpretation"),
        "kernel_pca": (25, "a nonlinear kernel comparison for moderate sample sizes"),
        "mds": (15, "direct pairwise-distance geometry"),
        "isomap": (20, "geodesic manifold structure"),
        "lle": (18, "locally linear manifold structure"),
        "spectral_embedding": (22, "graph-based neighborhood structure"),
        "tsne": (25, "local-neighborhood visualization"),
        "umap": (25, "local manifold visualization with scalable graph construction"),
    }
    if ratio >= 5:
        scores["pca"] = (75, "a stable linear baseline for a feature count much larger than n")
        scores["sparse_pca"] = (85, "sparse loadings suit a feature count much larger than n")
        scores["umap"] = (45, "local structure may remain after high-dimensional preprocessing")
    if profile.get("storage_type") == "sparse":
        scores["truncated_svd"] = (100, "the sparse matrix can be reduced without centering or root densification")
    if n_samples <= 250 and n_features <= 100:
        scores["mds"] = (70, "the small sample count makes pairwise-distance geometry feasible")
        scores["kernel_pca"] = (60, "the sample count permits a nonlinear kernel comparison")
    if 50 <= n_samples <= 2_000:
        scores["isomap"] = (58, "the sample count permits a geodesic-neighborhood comparison")
        scores["spectral_embedding"] = (56, "the sample count permits graph spectral analysis")
        scores["lle"] = (52, "the sample count permits a locally linear comparison")
        scores["tsne"] = (50, "the sample count permits a local visualization comparison")
    if n_samples >= 1_000:
        scores["umap"] = (80, "graph-based local exploration is preferable to quadratic methods at larger n")
        scores["pca"] = (65, "an efficient linear baseline remains useful at larger n")

    ranked = [
        (method, score, rationale)
        for method, (score, rationale) in scores.items()
        if feasibility.get(method, {}).get("feasible")
    ]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def default_method_parameters(
    method: str,
    profile: dict[str, Any],
    n_components: int = 2,
) -> dict[str, Any]:
    n_samples = int(profile.get("n_samples") or 2)
    neighbors = max(n_components + 1 if method == "lle" else 2, min(15, n_samples - 1))
    defaults: dict[str, dict[str, Any]] = {
        "truncated_svd": {"algorithm": "randomized", "n_iter": 5},
        "pca": {"svd_solver": "auto"},
        "sparse_pca": {"alpha": 1.0, "max_iter": 1000},
        "kernel_pca": {"kernel": "rbf"},
        "mds": {"n_init": 1, "max_iter": 300, "eps": 1e-3},
        "isomap": {"n_neighbors": neighbors},
        "lle": {"n_neighbors": neighbors, "method": "standard"},
        "spectral_embedding": {"n_neighbors": neighbors, "affinity": "nearest_neighbors"},
        "tsne": {"perplexity": max(1.0, min(30.0, (n_samples - 1) / 3)), "max_iter": 1000},
        "umap": {"n_neighbors": neighbors, "min_dist": 0.1},
    }
    return defaults[method]


def assess_method_feasibility(
    method: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    capability = METHOD_CAPABILITIES.get(method)
    if capability is None:
        return {"feasible": False, "reasons": [f"Unknown method: {method}"]}
    reasons: list[str] = []
    if not capability.executable:
        reasons.append("Method is not executable in the current version.")
    storage_type = profile.get("storage_type", "dense")
    if storage_type == "sparse" and not capability.sparse_input:
        reasons.append("Current DeEntropy execution path does not support sparse input for this method.")
    if storage_type == "dense" and not capability.dense_input:
        reasons.append("Current DeEntropy execution path does not support dense input for this method.")
    n_samples = int(profile.get("n_samples") or 0)
    n_features = int(profile.get("n_features") or 0)
    if n_samples < 2 or n_features < 1:
        reasons.append("At least two samples and one feature are required.")
    if capability.quadratic_sample_scaling and n_samples > QUADRATIC_SAMPLE_LIMIT:
        reasons.append(
            f"Quadratic sample scaling is disabled above {QUADRATIC_SAMPLE_LIMIT} samples."
        )
    if method in {"isomap", "lle", "spectral_embedding", "umap"} and n_samples < 4:
        reasons.append("Neighborhood methods require at least four samples.")
    if method == "tsne" and n_samples < 3:
        reasons.append("t-SNE requires at least three samples.")
    return {"feasible": not reasons, "reasons": reasons}


def validate_method_plan(
    method: str,
    n_components: int,
    parameters: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    capability = METHOD_CAPABILITIES.get(method)
    if capability is None or not capability.executable:
        raise ValueError(f"Method '{method}' is not executable.")
    feasibility = assess_method_feasibility(method, profile)
    if not feasibility["feasible"]:
        raise ValueError(f"Method '{method}' is infeasible: {'; '.join(feasibility['reasons'])}")

    n_samples = int(profile.get("n_samples") or 0)
    n_features = int(profile.get("n_features") or 0)
    if n_components < 1 or n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components={n_components} must be between 1 and min(n_samples, n_features)."
        )
    if method in {"isomap", "lle", "spectral_embedding", "tsne", "umap"} and n_components >= n_samples:
        raise ValueError(f"{method} requires n_components smaller than n_samples.")
    if method == "pca" and parameters.get("svd_solver") == "arpack" and n_components >= min(n_samples, n_features):
        raise ValueError("PCA with svd_solver='arpack' requires n_components below the matrix rank bound.")
    if method == "truncated_svd" and n_components >= n_features:
        raise ValueError("Truncated SVD requires n_components smaller than n_features.")

    unknown = set(parameters) - set(capability.parameters)
    if unknown:
        raise ValueError(
            f"Unsupported parameters for {method}: {', '.join(sorted(unknown))}."
        )
    for name, value in parameters.items():
        _validate_parameter(method, name, value, capability.parameters[name])

    neighbors = parameters.get("n_neighbors")
    if neighbors is not None:
        if int(neighbors) >= n_samples:
            raise ValueError(f"{method} n_neighbors must be smaller than n_samples.")
        if method == "lle" and int(neighbors) <= n_components:
            raise ValueError("LLE n_neighbors must be greater than n_components.")
        if method == "lle" and parameters.get("method") == "hessian":
            minimum_hessian_neighbors = n_components * (n_components + 3) // 2 + 1
            if int(neighbors) < minimum_hessian_neighbors:
                raise ValueError(
                    "Hessian LLE requires n_neighbors greater than "
                    "n_components * (n_components + 3) / 2."
                )
    if method == "tsne":
        perplexity = float(parameters.get("perplexity", min(30.0, (n_samples - 1) / 3)))
        if perplexity <= 0 or perplexity >= n_samples:
            raise ValueError("t-SNE perplexity must be positive and smaller than n_samples.")
    if method == "umap":
        min_dist = float(parameters.get("min_dist", 0.1))
        spread = float(parameters.get("spread", 1.0))
        if spread <= 0 or min_dist > spread:
            raise ValueError("UMAP requires spread > 0 and min_dist <= spread.")
    return parameters


def _validate_parameter(
    method: str,
    name: str,
    value: Any,
    rule: ParameterCapability,
) -> None:
    if value is None and "none" in rule.value_type:
        return
    if rule.choices and value not in rule.choices:
        raise ValueError(f"Invalid {method} {name}={value!r}; allowed: {rule.choices}.")
    numeric_types = {"number", "number_or_none", "number_or_auto"}
    integer_types = {"integer", "integer_or_none", "integer_or_auto"}
    if rule.value_type in numeric_types and value != "auto":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{method} {name} must be numeric.")
    if rule.value_type in integer_types and value not in {"auto", None}:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{method} {name} must be an integer.")
    if rule.value_type == "boolean" and not isinstance(value, bool):
        raise ValueError(f"{method} {name} must be boolean.")
    is_special = value == "auto" or value is None
    if rule.minimum is not None and not is_special and value < rule.minimum:
        raise ValueError(f"{method} {name} must be at least {rule.minimum}.")
    if rule.maximum is not None and not is_special and value > rule.maximum:
        raise ValueError(f"{method} {name} must be at most {rule.maximum}.")
