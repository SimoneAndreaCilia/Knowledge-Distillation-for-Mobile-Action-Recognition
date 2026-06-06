# -*- coding: utf-8 -*-
"""
Comparison Visualization Module
=================================

Generates publication-quality figures comparing Teacher, Baseline Student,
and Distilled Student performance. All figures are saved to disk and can
be logged to TensorBoard.

Produces the following figure types:

  1. **Accuracy Bar Chart**: Side-by-side top-1 test accuracy comparison.
  2. **Deployment Dashboard**: Multi-panel view of params, size, latency,
     and accuracy trade-offs.
  3. **Per-Class Accuracy Heatmap**: Highlights classes where KD helps
     or hurts relative to the baseline.
  4. **Temperature Ablation Plot**: Accuracy vs. temperature curve.
  5. **Compression vs. Accuracy Scatter**: Pareto frontier visualization.

Usage::

    from src.evaluation.comparison import (
        plot_accuracy_comparison,
        plot_deployment_dashboard,
        plot_per_class_heatmap,
        plot_temperature_ablation,
        save_all_figures,
    )

    results = {
        "Teacher": {"top1_acc": 65.2, "param_count": 46e6, ...},
        "Baseline": {"top1_acc": 42.1, "param_count": 2.4e6, ...},
        "Distilled": {"top1_acc": 55.3, "param_count": 2.4e6, ...},
    }

    figs = save_all_figures(results, output_dir="./figures")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# COLOR PALETTE & STYLE
# ======================================================================

# Consistent palette: Teacher=blue, Baseline=orange, Distilled=green
MODEL_COLORS = {
    "Teacher (ResNet3D-50)": "#2563EB",   # Blue
    "Student Baseline":      "#F59E0B",   # Amber
    "Student Distilled":     "#10B981",   # Emerald
}

# Fallback colors for arbitrary model names
FALLBACK_COLORS = [
    "#2563EB", "#F59E0B", "#10B981", "#EF4444",
    "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16",
]


def _get_color(name: str, idx: int = 0) -> str:
    """Get a consistent color for a model name."""
    return MODEL_COLORS.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def _apply_style(fig, ax_or_axes):
    """Apply a clean, publication-ready style to a figure."""
    import matplotlib.pyplot as plt
    fig.patch.set_facecolor("#FAFAFA")
    axes = ax_or_axes if hasattr(ax_or_axes, "__iter__") else [ax_or_axes]
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.tick_params(colors="#555555")
        ax.xaxis.label.set_color("#333333")
        ax.yaxis.label.set_color("#333333")


# ======================================================================
# FIGURE 1: ACCURACY BAR CHART
# ======================================================================

def plot_accuracy_comparison(
    results: Dict[str, Dict[str, Any]],
    title: str = "Test Accuracy Comparison",
    figsize: Tuple[int, int] = (10, 6),
) -> "matplotlib.figure.Figure":
    """Create a grouped bar chart comparing top-1 test accuracies.

    Shows accuracy bars with exact percentage labels on top.
    Includes a horizontal reference line at the Teacher's accuracy
    to visually gauge the KD gap closure.

    Args:
        results: Dict mapping model names to result dicts (must have ``"top1_acc"``).
        title: Figure title.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    names = list(results.keys())
    accs = [results[n]["top1_acc"] for n in names]
    colors = [_get_color(n, i) for i, n in enumerate(names)]

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(fig, ax)

    bars = ax.bar(names, accs, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    # Value labels on top of bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#333333",
        )

    # Teacher reference line (if Teacher present)
    teacher_acc = None
    for name in names:
        if "teacher" in name.lower():
            teacher_acc = results[name]["top1_acc"]
            break

    if teacher_acc is not None:
        ax.axhline(
            y=teacher_acc, color="#2563EB", linestyle="--",
            alpha=0.4, linewidth=1.5, label=f"Teacher: {teacher_acc:.1f}%",
        )
        ax.legend(fontsize=10, loc="lower right")

    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, max(accs) * 1.15)

    plt.tight_layout()
    return fig


# ======================================================================
# FIGURE 2: DEPLOYMENT DASHBOARD (4-panel)
# ======================================================================

def plot_deployment_dashboard(
    results: Dict[str, Dict[str, Any]],
    title: str = "Deployment Metrics Dashboard",
    figsize: Tuple[int, int] = (16, 10),
) -> "matplotlib.figure.Figure":
    """Create a 2×2 deployment comparison dashboard.

    Panels:
      1. **Accuracy** — bar chart of test accuracies.
      2. **Parameters** — bar chart with parameter counts (log scale).
      3. **Model Size** — bar chart in MB.
      4. **Inference Latency** — bar chart in ms (if available).

    Args:
        results: Dict mapping model names to result dicts.
        title: Dashboard suptitle.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    names = list(results.keys())
    colors = [_get_color(n, i) for i, n in enumerate(names)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    _apply_style(fig, axes.flat)

    # Panel 1: Accuracy
    ax = axes[0, 0]
    accs = [results[n].get("top1_acc", 0) for n in names]
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test Accuracy", fontweight="bold")
    ax.set_ylim(0, max(accs) * 1.15 if accs else 100)

    # Panel 2: Parameters
    ax = axes[0, 1]
    params = [results[n].get("param_count", 0) for n in names]
    params_m = [p / 1e6 for p in params]
    bars = ax.bar(names, params_m, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, params_m):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}M", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Parameters (millions)")
    ax.set_title("Model Complexity", fontweight="bold")

    # Panel 3: Model Size
    ax = axes[1, 0]
    sizes = [results[n].get("model_size_mb", 0) for n in names]
    bars = ax.bar(names, sizes, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f} MB", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Size (MB)")
    ax.set_title("Model Size (Memory)", fontweight="bold")

    # Panel 4: Latency
    ax = axes[1, 1]
    latencies = [results[n].get("latency_ms", 0) for n in names]
    if any(l > 0 for l in latencies):
        bars = ax.bar(names, latencies, color=colors, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, latencies):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{val:.1f} ms", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Inference Latency", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Latency not measured",
                ha="center", va="center", fontsize=12, color="#999999",
                transform=ax.transAxes)
        ax.set_title("Inference Latency", fontweight="bold")

    # Rotate x-labels for readability
    for a in axes.flat:
        a.tick_params(axis="x", rotation=15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ======================================================================
# FIGURE 3: PER-CLASS ACCURACY HEATMAP
# ======================================================================

def plot_per_class_heatmap(
    results: Dict[str, Dict[str, Any]],
    class_names: Optional[List[str]] = None,
    top_k: int = 20,
    figsize: Tuple[int, int] = (14, 8),
) -> "matplotlib.figure.Figure":
    """Create a per-class accuracy heatmap comparing models.

    Shows the top-K most interesting classes (largest accuracy
    difference between models) to keep the figure readable.

    Args:
        results: Dict with ``"per_class_acc"`` lists.
        class_names: List of class name strings.
        top_k: Number of classes to display (most variable).
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = list(results.keys())
    n_models = len(names)

    # Extract per-class accuracies
    per_class = {}
    for name in names:
        pca = results[name].get("per_class_acc", [])
        if pca:
            per_class[name] = np.array(pca)

    if not per_class:
        logger.warning("No per-class accuracy data available for heatmap.")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Per-class data not available",
                ha="center", va="center", fontsize=14)
        return fig

    # Stack into matrix: (n_models, n_classes)
    all_pca = np.stack([per_class[n] for n in names if n in per_class])
    n_classes = all_pca.shape[1]

    # Select top-K most variable classes (most "interesting" for comparison)
    variance = all_pca.var(axis=0)
    top_indices = np.argsort(variance)[-top_k:][::-1]

    # Prepare labels
    if class_names is None:
        class_labels = [f"Class {i}" for i in top_indices]
    else:
        class_labels = [class_names[i] if i < len(class_names) else f"Class {i}"
                        for i in top_indices]

    # Extract the subset
    data = all_pca[:, top_indices]  # (n_models, top_k)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(fig, ax)

    # Custom colormap: red (0%) → white (50%) → green (100%)
    cmap = LinearSegmentedColormap.from_list(
        "accuracy", ["#EF4444", "#FEFCE8", "#10B981"]
    )

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    # Axis labels
    model_names_in = [n for n in names if n in per_class]
    ax.set_yticks(range(len(model_names_in)))
    ax.set_yticklabels(model_names_in, fontsize=10)
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)

    # Annotate cells with values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] < 30 or data[i, j] > 85 else "black"
            ax.text(j, i, f"{data[i, j]:.0f}",
                    ha="center", va="center", fontsize=8, color=color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=10)

    ax.set_title(
        f"Per-Class Accuracy (Top {top_k} Most Variable Classes)",
        fontsize=13, fontweight="bold", pad=15,
    )

    plt.tight_layout()
    return fig


# ======================================================================
# FIGURE 4: TEMPERATURE ABLATION
# ======================================================================

def plot_temperature_ablation(
    temp_results: Dict[float, float],
    baseline_acc: Optional[float] = None,
    teacher_acc: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> "matplotlib.figure.Figure":
    """Plot accuracy vs. temperature for the KD ablation study.

    Shows how different temperature values affect the distilled
    Student's accuracy, with optional reference lines for the
    Teacher (upper bound) and Baseline Student (lower bound).

    Args:
        temp_results: Dict mapping temperature values to accuracy.
            Example: ``{1: 48.5, 5: 55.3, 10: 53.1, 20: 50.2}``.
        baseline_acc: Baseline Student accuracy (horizontal reference).
        teacher_acc: Teacher accuracy (horizontal reference).
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    temps = sorted(temp_results.keys())
    accs = [temp_results[t] for t in temps]

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(fig, ax)

    # Main line with markers
    ax.plot(
        temps, accs,
        color="#10B981", linewidth=2.5, marker="o", markersize=10,
        markerfacecolor="white", markeredgewidth=2.5,
        markeredgecolor="#10B981", zorder=5,
    )

    # Annotate each point
    for t, a in zip(temps, accs):
        ax.annotate(
            f"{a:.1f}%",
            (t, a),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center", fontsize=11, fontweight="bold",
            color="#10B981",
        )

    # Reference lines
    if teacher_acc is not None:
        ax.axhline(
            y=teacher_acc, color="#2563EB", linestyle="--",
            alpha=0.6, linewidth=1.5,
        )
        ax.text(
            temps[-1], teacher_acc + 0.5,
            f"Teacher: {teacher_acc:.1f}%",
            ha="right", fontsize=10, color="#2563EB",
        )

    if baseline_acc is not None:
        ax.axhline(
            y=baseline_acc, color="#F59E0B", linestyle="--",
            alpha=0.6, linewidth=1.5,
        )
        ax.text(
            temps[-1], baseline_acc - 1.5,
            f"Baseline: {baseline_acc:.1f}%",
            ha="right", fontsize=10, color="#F59E0B",
        )

    # Shade the "improvement region" between baseline and distilled
    if baseline_acc is not None:
        ax.fill_between(
            temps,
            [baseline_acc] * len(temps),
            accs,
            alpha=0.08,
            color="#10B981",
            where=[a >= baseline_acc for a in accs],
        )

    ax.set_xlabel("Temperature (T)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Temperature Ablation — KD Accuracy vs. T",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xticks(temps)

    plt.tight_layout()
    return fig


# ======================================================================
# FIGURE 5: COMPRESSION vs ACCURACY SCATTER
# ======================================================================

def plot_compression_scatter(
    results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (10, 7),
) -> "matplotlib.figure.Figure":
    """Plot the accuracy vs. model size trade-off (Pareto frontier).

    Each model is a point; the axes show model size (MB) on X and
    test accuracy (%) on Y. This visualizes how much accuracy is
    retained per unit of compression.

    Args:
        results: Dict mapping model names to result dicts.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(fig, ax)

    for i, (name, res) in enumerate(results.items()):
        size = res.get("model_size_mb", 0)
        acc = res.get("top1_acc", 0)
        color = _get_color(name, i)

        ax.scatter(
            size, acc, s=200, c=color, edgecolors="white",
            linewidth=2, zorder=5, label=name,
        )
        ax.annotate(
            f"{name}\n{acc:.1f}%\n{size:.0f} MB",
            (size, acc),
            textcoords="offset points",
            xytext=(15, 5),
            fontsize=9, color=color,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    ax.set_xlabel("Model Size (MB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Accuracy vs. Model Size Trade-off",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    return fig


# ======================================================================
# SAVE ALL FIGURES
# ======================================================================

def save_all_figures(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "./figures",
    class_names: Optional[List[str]] = None,
    temp_results: Optional[Dict[float, float]] = None,
    dpi: int = 150,
) -> Dict[str, "matplotlib.figure.Figure"]:
    """Generate and save all comparison figures to disk.

    Args:
        results: Full model comparison results dict.
        output_dir: Directory for saved figure files.
        class_names: Optional class name list for heatmaps.
        temp_results: Optional temperature ablation data.
        dpi: Resolution for saved figures.

    Returns:
        Dict mapping figure names to Figure objects (for TensorBoard).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    figures = {}

    # 1. Accuracy bar chart
    fig = plot_accuracy_comparison(results)
    fig.savefig(out / "accuracy_comparison.png", dpi=dpi, bbox_inches="tight")
    figures["accuracy_comparison"] = fig
    logger.info(f"Saved: {out / 'accuracy_comparison.png'}")

    # 2. Deployment dashboard
    fig = plot_deployment_dashboard(results)
    fig.savefig(out / "deployment_dashboard.png", dpi=dpi, bbox_inches="tight")
    figures["deployment_dashboard"] = fig
    logger.info(f"Saved: {out / 'deployment_dashboard.png'}")

    # 3. Per-class heatmap
    has_per_class = any("per_class_acc" in r for r in results.values())
    if has_per_class:
        fig = plot_per_class_heatmap(results, class_names=class_names)
        fig.savefig(out / "per_class_heatmap.png", dpi=dpi, bbox_inches="tight")
        figures["per_class_heatmap"] = fig
        logger.info(f"Saved: {out / 'per_class_heatmap.png'}")

    # 4. Compression scatter
    fig = plot_compression_scatter(results)
    fig.savefig(out / "compression_scatter.png", dpi=dpi, bbox_inches="tight")
    figures["compression_scatter"] = fig
    logger.info(f"Saved: {out / 'compression_scatter.png'}")

    # 5. Temperature ablation (if data provided)
    if temp_results:
        teacher_acc = None
        baseline_acc = None
        for name, res in results.items():
            if "teacher" in name.lower():
                teacher_acc = res.get("top1_acc")
            elif "baseline" in name.lower():
                baseline_acc = res.get("top1_acc")

        fig = plot_temperature_ablation(
            temp_results, baseline_acc=baseline_acc, teacher_acc=teacher_acc
        )
        fig.savefig(out / "temperature_ablation.png", dpi=dpi, bbox_inches="tight")
        figures["temperature_ablation"] = fig
        logger.info(f"Saved: {out / 'temperature_ablation.png'}")

    logger.info(f"All {len(figures)} figures saved to {output_dir}/")
    return figures
