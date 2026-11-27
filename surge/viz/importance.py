"""Feature importance analysis using SHAP (SHapley Additive exPlanations).

Provides TreeExplainer (fast, for tree models) and KernelExplainer (model-agnostic).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


def _infer_feature_names(
    X: Any,
    feature_names: Optional[List[str]],
    n_features: int,
    *,
    data_source: Optional[Any] = None,
) -> List[str]:
    """Infer feature names when not explicitly provided.

    Priority:
    1. feature_names if given
    2. data_source.input_columns or data_source._input_columns or data_source.input_feature_names
    3. X.columns if X is a DataFrame
    4. input_0, input_1, ...
    """
    if feature_names is not None:
        return list(feature_names)
    if data_source is not None:
        for attr in ("input_columns", "_input_columns", "input_feature_names"):
            cols = getattr(data_source, attr, None)
            if cols is not None and len(cols) == n_features:
                return list(cols)
    if PANDAS_AVAILABLE and hasattr(X, "columns") and hasattr(X, "iloc"):
        cols = getattr(X, "columns", None)
        if cols is not None and len(cols) == n_features:
            return list(cols)
    return [f"input_{i}" for i in range(n_features)]

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _get_predict_fn(model: Any, output_index: Optional[int] = None) -> Callable:
    """Get a predict function for the model (adapter or raw estimator)."""
    # SURGE adapter
    if hasattr(model, "predict"):
        pred_fn = model.predict
        if output_index is not None:

            def _predict(X):
                out = pred_fn(X)
                out = np.asarray(out)
                if out.ndim == 2 and out.shape[1] > 1:
                    return out[:, output_index]
                return out

            return _predict
        return pred_fn
    # Raw sklearn-style estimator
    return model.predict


def _get_tree_model(model: Any) -> Any:
    """Extract tree model from adapter if needed."""
    if hasattr(model, "_model"):
        return model._model
    return model


def _is_tree_model(model: Any) -> bool:
    """Check if model supports TreeExplainer (sklearn RF, XGBoost, etc.)."""
    try:
        tree = _get_tree_model(model)
        return hasattr(tree, "estimators_") or hasattr(tree, "get_booster")
    except Exception:
        return False


def compute_shap_tree(
    model: Any,
    X: np.ndarray,
    *,
    X_background: Optional[np.ndarray] = None,
    output_index: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    check_additivity: bool = True,
) -> Tuple[Any, np.ndarray]:
    """
    Compute SHAP values using TreeExplainer (fast for tree models).

    Parameters
    ----------
    model : object
        Trained tree model (RandomForest, XGBoost, etc.) or SURGE adapter.
    X : np.ndarray
        Data to explain, shape (n_samples, n_features).
    X_background : np.ndarray, optional
        Background dataset for TreeExplainer. If None, uses X (or first 100 rows).
    output_index : int, optional
        For multi-output, which output to explain (0-based).
    feature_names : list, optional
        Names for each feature.
    check_additivity : bool, default=True
        Whether to verify SHAP values sum to prediction difference.

    Returns
    -------
    explainer : shap.TreeExplainer
        The explainer object.
    shap_values : np.ndarray
        SHAP values, shape (n_samples, n_features).

    Raises
    ------
    ImportError
        If shap is not installed.
    ValueError
        If model is not a tree model.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required. Install with: pip install shap")

    tree_model = _get_tree_model(model)
    if not _is_tree_model(model):
        raise ValueError(
            "TreeExplainer requires a tree-based model (RandomForest, XGBoost, etc.). "
            "Use compute_shap_kernel for other models."
        )

    X = np.asarray(X, dtype=np.float64)
    if X_background is None:
        X_background = X[: min(100, len(X))]

    explainer = shap.TreeExplainer(
        tree_model,
        data=X_background,
        feature_perturbation="interventional",
    )

    # For multi-output, TreeExplainer may return list of arrays or 3D array
    shap_vals = explainer.shap_values(X, check_additivity=check_additivity)
    if isinstance(shap_vals, list):
        idx = output_index if output_index is not None else 0
        shap_vals = np.asarray(shap_vals[idx])
    else:
        shap_vals = np.asarray(shap_vals)
        if shap_vals.ndim == 3:
            # (n_samples, n_features, n_outputs) -> take output_index
            idx = output_index if output_index is not None else 0
            shap_vals = shap_vals[:, :, idx]

    return explainer, shap_vals


def compute_shap_kernel(
    model: Any,
    X: np.ndarray,
    X_background: np.ndarray,
    *,
    output_index: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    nsamples: int = 100,
    link: str = "identity",
) -> Tuple[Any, np.ndarray]:
    """
    Compute SHAP values using KernelExplainer (model-agnostic, slower).

    Parameters
    ----------
    model : object
        Trained model or SURGE adapter with predict(X) method.
    X : np.ndarray
        Data to explain, shape (n_samples, n_features).
    X_background : np.ndarray
        Background dataset (smaller is faster). Typically 50-500 samples.
    output_index : int, optional
        For multi-output, which output to explain (0-based).
    feature_names : list, optional
        Names for each feature.
    nsamples : int, default=100
        Number of evaluations per prediction. Higher = more accurate, slower.
    link : str, default="identity"
        "identity" or "logit" for classification.

    Returns
    -------
    explainer : shap.KernelExplainer
        The explainer object.
    shap_values : np.ndarray
        SHAP values, shape (n_samples, n_features).

    Raises
    ------
    ImportError
        If shap is not installed.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required. Install with: pip install shap")

    predict_fn = _get_predict_fn(model, output_index=output_index)
    X = np.asarray(X, dtype=np.float64)
    X_background = np.asarray(X_background, dtype=np.float64)

    explainer = shap.KernelExplainer(predict_fn, X_background, link=link)
    shap_vals = explainer.shap_values(X, nsamples=nsamples)

    if isinstance(shap_vals, list):
        shap_vals = np.asarray(shap_vals[0]) if shap_vals else np.zeros_like(X)
    else:
        shap_vals = np.asarray(shap_vals)

    return explainer, shap_vals


def shap_importance_mean_abs(shap_values: np.ndarray) -> np.ndarray:
    """Mean absolute SHAP value per feature (global importance)."""
    return np.abs(shap_values).mean(axis=0)


def shap_grouped_importance(
    shap_values: np.ndarray,
    group_specs: List[Tuple[str, List[int]]],
) -> Tuple[List[str], np.ndarray]:
    """
    Aggregate SHAP importance by feature groups.

    Parameters
    ----------
    shap_values : np.ndarray
        Shape (n_samples, n_features).
    group_specs : list of (name, indices)
        Each (name, indices) defines a group; indices are column indices.

    Returns
    -------
    names : list of str
        Group names.
    importance : np.ndarray
        Mean |SHAP| per group (sum over group, then mean over samples).
    """
    n_features = shap_values.shape[1]
    abs_sv = np.abs(shap_values)
    names = []
    vals = []
    for name, indices in group_specs:
        idx = [i for i in indices if 0 <= i < n_features]
        if idx:
            # Mean over samples of (sum of |SHAP| over group)
            group_imp = abs_sv[:, idx].sum(axis=1).mean()
            names.append(name)
            vals.append(float(group_imp))
    return names, np.array(vals)


def stage_group_specs(
    n_features: int,
    n_per_stage: int,
    n_extra: int = 0,
) -> List[Tuple[str, List[int]]]:
    """
    Group spec for stacked layouts: consecutive stages of ``n_per_stage`` inputs,
    optionally followed by ``n_extra`` trailing inputs as one extra group.

    ``n_features`` must equal ``n_stages * n_per_stage + n_extra``.
    """
    if n_per_stage < 1 or n_extra < 0:
        raise ValueError("n_per_stage must be >= 1 and n_extra >= 0")
    n_stack = n_features - n_extra
    if n_stack < 0 or n_stack % n_per_stage != 0:
        raise ValueError(
            f"n_features={n_features} is not n_stages * {n_per_stage} + {n_extra}"
        )
    specs: List[Tuple[str, List[int]]] = []
    for s in range(n_stack // n_per_stage):
        start, end = s * n_per_stage, (s + 1) * n_per_stage
        specs.append((f"stage_{s} (inputs {start}-{end-1})", list(range(start, end))))
    if n_extra:
        specs.append(
            (f"extra_{n_extra} (inputs {n_stack}-{n_features-1})", list(range(n_stack, n_features)))
        )
    return specs


def plot_shap_grouped_bar(
    shap_values: np.ndarray,
    group_specs: Optional[List[Tuple[str, List[int]]]] = None,
    *,
    title: str = "SHAP importance by group",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    figsize: Tuple[float, float] = (8, 8),
) -> Any:
    """
    Bar plot of mean |SHAP| per feature group.

    ``group_specs`` is required; build it with ``stage_group_specs`` for stacked layouts.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plot_shap_grouped_bar")
    if group_specs is None:
        raise ValueError("group_specs is required for plot_shap_grouped_bar")
    names, importance = shap_grouped_importance(shap_values, group_specs)
    order = np.argsort(importance)[::-1]
    names = [names[i] for i in order]
    importance = importance[order]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(names)), importance, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (group total)")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X: Union[np.ndarray, Any],
    *,
    feature_names: Optional[List[str]] = None,
    plot_type: str = "bar",
    max_display: int = 20,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    show: bool = False,
) -> Any:
    """
    Plot SHAP summary (bar or beeswarm).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values, shape (n_samples, n_features).
    X : np.ndarray or pandas.DataFrame
        Feature data, shape (n_samples, n_features). If DataFrame, feature names
        are inferred from X.columns when feature_names is None.
    feature_names : list, optional
        Names for each feature. If None, inferred from X.columns (DataFrame) or
        input_0, input_1, ... (array).
    plot_type : str, default="bar"
        "bar" for mean |SHAP|, "dot" for beeswarm.
    max_display : int, default=20
        Max features to show.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figure.
    show : bool, default=False
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure if plot_type="bar" (custom bar plot), else None (shap handles it).
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required. Install with: pip install shap")
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")

    X_arr = np.asarray(X, dtype=np.float64)
    feature_names = _infer_feature_names(X, feature_names, shap_values.shape[1])

    if plot_type == "bar":
        # Bar: mean |SHAP| per feature = global importance. Longer bar = more impact on predictions.
        importance = shap_importance_mean_abs(shap_values)
        indices = np.argsort(importance)[::-1][:max_display]
        names = [feature_names[i] for i in indices]
        vals = importance[indices]

        fig, ax = plt.subplots(figsize=(8, min(max_display * 0.4 + 1, 12)))
        ax.barh(range(len(names)), vals, color="steelblue", alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature importance (SHAP)")
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # beeswarm via shap
    # Beeswarm: each dot = one instance; x = SHAP value (impact); color = feature value (red=high, blue=low).
    # Shows how feature values correlate with their impact across all samples.
    shap.summary_plot(
        shap_values,
        X_arr,
        feature_names=feature_names,
        plot_type="dot",
        max_display=max_display,
        show=show,
    )
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_ind: Union[int, str],
    *,
    feature_names: Optional[List[str]] = None,
    interaction_index: Optional[Union[int, str]] = "auto",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    show: bool = False,
) -> Any:
    """
    Dependence plot: how a single feature's value affects its SHAP values.

    Plots feature value (x-axis) vs SHAP value (y-axis). Shows the marginal effect
    of the feature on the model output. Vertical dispersion indicates interaction
    with other features. Points are colored by an interacting feature (by default,
    the one with strongest interaction) to reveal dependencies.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values, shape (n_samples, n_features).
    X : np.ndarray or pandas.DataFrame
        Feature data. If DataFrame, feature names inferred from X.columns when None.
    feature_ind : int or str
        Index or name of the feature to plot. Use "rank(k)" for k-th most important.
    feature_names : list, optional
        Names for each feature.
    interaction_index : int, str, or None, default="auto"
        Feature to color points by (interaction). "auto" picks strongest interaction.
        None disables interaction coloring.
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figure.
    show : bool, default=False
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required. Install with: pip install shap")
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")

    shap_values = np.asarray(shap_values)
    X_arr = np.asarray(X, dtype=np.float64)
    feature_names = _infer_feature_names(X, feature_names, shap_values.shape[1])
    plt.figure()
    shap.dependence_plot(
        feature_ind,
        shap_values,
        X_arr,
        feature_names=feature_names,
        interaction_index=interaction_index,
        show=show,
    )
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_shap_waterfall(
    explainer: Any,
    shap_values: np.ndarray,
    X: np.ndarray,
    sample_index: int = 0,
    *,
    feature_names: Optional[List[str]] = None,
    max_display: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    show: bool = False,
) -> Any:
    """
    Waterfall plot: step-by-step explanation of a single prediction.

    Shows how each feature's SHAP value moves the model output from the base
    (expected) value to the final prediction. Bars represent each feature's
    contribution; positive values push the prediction up, negative push down.
    Features are sorted by |SHAP| magnitude. Use to understand "why did the
    model predict this value for this specific instance?"

    Parameters
    ----------
    explainer : shap.Explainer
        Fitted explainer (TreeExplainer or KernelExplainer) with expected_value.
    shap_values : np.ndarray
        SHAP values, shape (n_samples, n_features).
    X : np.ndarray or pandas.DataFrame
        Feature data. If DataFrame, feature names inferred from X.columns when None.
    sample_index : int, default=0
        Which sample to explain.
    feature_names : list, optional
        Names for each feature.
    max_display : int, default=10
        Max features to show; rest grouped as "other".
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figure.
    show : bool, default=False
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required. Install with: pip install shap")
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")

    X_arr = np.asarray(X, dtype=np.float64)
    feature_names = _infer_feature_names(X, feature_names, shap_values.shape[1])

    ev = getattr(explainer, "expected_value", None)
    if ev is None:
        raise ValueError("Explainer must have expected_value attribute.")
    if isinstance(ev, (list, np.ndarray)):
        ev = float(ev[0]) if len(ev) > 0 else 0.0
    else:
        ev = float(ev)

    sv = np.asarray(shap_values[sample_index])
    x_row = np.asarray(X_arr[sample_index])
    exp = shap.Explanation(
        values=sv,
        base_values=ev,
        data=x_row,
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=show)
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_shap_force(
    explainer: Any,
    shap_values: np.ndarray,
    X: np.ndarray,
    sample_index: int = 0,
    *,
    feature_names: Optional[List[str]] = None,
    contribution_threshold: float = 0.05,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    show: bool = False,
) -> Any:
    """
    Force plot: additive force layout for a single prediction.

    Visualizes how features push the model output from the base value to the
    final prediction. Red arrows = push up; blue = push down. Bar length =
    magnitude of contribution. Use to quickly see which features drove a
    specific prediction and in which direction.

    Parameters
    ----------
    explainer : shap.Explainer
        Fitted explainer with expected_value.
    shap_values : np.ndarray
        SHAP values, shape (n_samples, n_features).
    X : np.ndarray or pandas.DataFrame
        Feature data. If DataFrame, feature names inferred from X.columns when None.
    sample_index : int, default=0
        Which sample to explain.
    feature_names : list, optional
        Names for each feature.
    contribution_threshold : float, default=0.05
        Only show features with |SHAP| > threshold * sum(|SHAP|).
    save_path : str or Path, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figure.
    show : bool, default=False
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure when matplotlib=True; None for HTML output.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required. Install with: pip install shap")
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")

    X_arr = np.asarray(X, dtype=np.float64)
    feature_names = _infer_feature_names(X, feature_names, shap_values.shape[1])

    ev = getattr(explainer, "expected_value", None)
    if ev is None:
        raise ValueError("Explainer must have expected_value attribute.")
    if isinstance(ev, (list, np.ndarray)):
        ev = float(ev[0]) if len(ev) > 0 else 0.0
    else:
        ev = float(ev)

    sv = np.asarray(shap_values[sample_index])
    x_row = np.asarray(X_arr[sample_index])
    plt.figure()
    shap.plots.force(
        ev,
        sv,
        x_row,
        feature_names=feature_names,
        matplotlib=True,
        show=show,
        contribution_threshold=contribution_threshold,
    )
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def run_shap_analysis(
    model: Any,
    X: Union[np.ndarray, Any],
    *,
    X_background: Optional[Union[np.ndarray, Any]] = None,
    output_index: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    data_source: Optional[Any] = None,
    use_tree: Optional[bool] = None,
    use_kernel: bool = True,
    kernel_nsamples: int = 50,
    save_dir: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    plot_type: str = "bar",
    save_dependence: bool = False,
    save_waterfall: bool = False,
    save_force: bool = False,
    dependence_features: Optional[List[Union[int, str]]] = None,
    waterfall_sample: int = 0,
    force_sample: int = 0,
    group_specs: Optional[List[Tuple[str, List[int]]]] = None,
) -> Dict[str, Any]:
    """
    Run SHAP analysis with TreeExplainer and/or KernelExplainer.

    Parameters
    ----------
    model : object
        Trained model or SURGE adapter.
    X : np.ndarray or pandas.DataFrame
        Data to explain. If DataFrame, feature names are inferred from X.columns
        when feature_names and data_source are not provided.
    X_background : np.ndarray or DataFrame, optional
        Background for explainers. If None, uses first 100 rows of X.
    output_index : int, optional
        For multi-output, which output to explain.
    feature_names : list, optional
        Feature names. If None, inferred from data_source, X.columns, or input_0, ...
    data_source : object, optional
        Engine, dataset, or object with input_columns/_input_columns/input_feature_names
        to infer feature names when feature_names is None.
    group_specs : list of (label, indices), optional
        If given, also save a grouped importance bar plot for the tree explainer
        (see ``stage_group_specs``). Not drawn by default.
    use_tree : bool, optional
        Use TreeExplainer if model is tree-based. If None, auto-detect.
    use_kernel : bool, default=True
        Use KernelExplainer (model-agnostic).
    kernel_nsamples : int, default=50
        nsamples for KernelExplainer (lower = faster).
    save_dir : str or Path, optional
        Directory to save plots.
    dpi : int, default=150
        Plot resolution.
    plot_type : str, default="bar"
        Summary plot: "bar" (mean |SHAP|), "dot" (beeswarm), or "both".
    save_dependence : bool, default=False
        Save dependence plots for top features.
    save_waterfall : bool, default=False
        Save waterfall plot for one sample.
    save_force : bool, default=False
        Save force plot for one sample.
    dependence_features : list, optional
        Feature indices/names for dependence plots. If None and save_dependence,
        uses top 3 by importance.
    waterfall_sample : int, default=0
        Sample index for waterfall plot.
    force_sample : int, default=0
        Sample index for force plot.

    Returns
    -------
    results : dict
        Keys: "tree" (if used), "kernel" (if used), "importance_tree", "importance_kernel",
        "saved_paths".
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required. Install with: pip install shap")

    X_arr = np.asarray(X, dtype=np.float64)
    n_features = X_arr.shape[1]
    feature_names = _infer_feature_names(X, feature_names, n_features, data_source=data_source)

    if X_background is None:
        X_background = X_arr[: min(100, len(X_arr))]
    else:
        X_background = np.asarray(X_background, dtype=np.float64)

    X = X_arr

    if plot_type not in ("bar", "dot", "both"):
        raise ValueError("plot_type must be 'bar', 'dot', or 'both'")

    results: Dict[str, Any] = {"saved_paths": []}
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # TreeExplainer
    if use_tree is None:
        use_tree = _is_tree_model(model)
    if use_tree:
        try:
            explainer_tree, shap_tree = compute_shap_tree(
                model, X,
                X_background=X_background,
                output_index=output_index,
                feature_names=feature_names,
            )
            results["tree"] = {"explainer": explainer_tree, "shap_values": shap_tree}
            results["importance_tree"] = shap_importance_mean_abs(shap_tree)
            if save_dir:
                for pt in (["bar", "dot"] if plot_type == "both" else [plot_type]):
                    path = save_dir / f"shap_summary_tree_{pt}.png"
                    try:
                        plot_shap_summary(
                            shap_tree, X,
                            feature_names=feature_names,
                            plot_type=pt,
                            save_path=path,
                            dpi=dpi,
                        )
                        results["saved_paths"].append(str(path))
                    except Exception as e:
                        if pt == "dot":
                            results["tree_error"] = str(results.get("tree_error", "")) + f"; dot plot: {e}"
                        else:
                            raise
                # Grouped SHAP only when the caller supplies a grouping
                if group_specs is not None:
                    try:
                        grouped_path = save_dir / "shap_summary_tree_grouped.png"
                        out_name = f"output_{output_index if output_index is not None else 0}"
                        plot_shap_grouped_bar(
                            shap_tree,
                            group_specs=group_specs,
                            title=f"SHAP importance by group ({out_name})",
                            save_path=grouped_path,
                            dpi=dpi,
                        )
                        results["saved_paths"].append(str(grouped_path))
                    except Exception as e:
                        results["tree_error"] = str(results.get("tree_error", "")) + f"; grouped: {e}"
                if save_dependence:
                    dep_feats = dependence_features or list(
                        np.argsort(results["importance_tree"])[::-1][:3]
                    )
                    for i, feat_ind in enumerate(dep_feats):
                        path = save_dir / f"shap_dependence_tree_{i}.png"
                        try:
                            plot_shap_dependence(
                                shap_tree, X, feat_ind,
                                feature_names=feature_names,
                                save_path=path,
                                dpi=dpi,
                            )
                            results["saved_paths"].append(str(path))
                        except Exception as e:
                            results["tree_error"] = str(results.get("tree_error", "")) + f"; dep_{i}: {e}"
                if save_waterfall:
                    path = save_dir / "shap_waterfall_tree.png"
                    try:
                        plot_shap_waterfall(
                            explainer_tree, shap_tree, X,
                            sample_index=waterfall_sample,
                            feature_names=feature_names,
                            save_path=path,
                            dpi=dpi,
                        )
                        results["saved_paths"].append(str(path))
                    except Exception as e:
                        results["tree_error"] = str(results.get("tree_error", "")) + f"; waterfall: {e}"
                if save_force:
                    path = save_dir / "shap_force_tree.png"
                    plot_shap_force(
                        explainer_tree, shap_tree, X,
                        sample_index=force_sample,
                        feature_names=feature_names,
                        save_path=path,
                        dpi=dpi,
                    )
                    results["saved_paths"].append(str(path))
        except Exception as e:
            results["tree_error"] = str(e)

    # KernelExplainer
    if use_kernel:
        try:
            bg = X_background[: min(50, len(X_background))]
            explainer_kernel, shap_kernel = compute_shap_kernel(
                model, X,
                X_background=bg,
                output_index=output_index,
                feature_names=feature_names,
                nsamples=kernel_nsamples,
            )
            results["kernel"] = {"explainer": explainer_kernel, "shap_values": shap_kernel}
            results["importance_kernel"] = shap_importance_mean_abs(shap_kernel)
            if save_dir:
                for pt in (["bar", "dot"] if plot_type == "both" else [plot_type]):
                    path = save_dir / f"shap_summary_kernel_{pt}.png"
                    plot_shap_summary(
                        shap_kernel, X,
                        feature_names=feature_names,
                        plot_type=pt,
                        save_path=path,
                        dpi=dpi,
                    )
                    results["saved_paths"].append(str(path))
                if save_dependence:
                    dep_feats = dependence_features or list(
                        np.argsort(results["importance_kernel"])[::-1][:3]
                    )
                    for i, feat_ind in enumerate(dep_feats):
                        path = save_dir / f"shap_dependence_kernel_{i}.png"
                        plot_shap_dependence(
                            shap_kernel, X, feat_ind,
                            feature_names=feature_names,
                            save_path=path,
                            dpi=dpi,
                        )
                        results["saved_paths"].append(str(path))
                if save_waterfall:
                    path = save_dir / "shap_waterfall_kernel.png"
                    plot_shap_waterfall(
                        explainer_kernel, shap_kernel, X,
                        sample_index=waterfall_sample,
                        feature_names=feature_names,
                        save_path=path,
                        dpi=dpi,
                    )
                    results["saved_paths"].append(str(path))
                if save_force:
                    path = save_dir / "shap_force_kernel.png"
                    plot_shap_force(
                        explainer_kernel, shap_kernel, X,
                        sample_index=force_sample,
                        feature_names=feature_names,
                        save_path=path,
                        dpi=dpi,
                    )
                    results["saved_paths"].append(str(path))
        except Exception as e:
            results["kernel_error"] = str(e)

    return results
