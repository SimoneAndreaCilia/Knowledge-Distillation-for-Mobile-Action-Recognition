# -*- coding: utf-8 -*-
"""
t-SNE Latent Space Visualization
==================================

Utility for visualizing how Teacher, Baseline Student, and Distilled Student
cluster actions in their penultimate-layer feature spaces.

Uses ``sklearn.manifold.TSNE`` to project high-dimensional feature vectors
(Teacher: 2048-D, Student: 1280-D) into 2D for plotting.

**Workflow**:
  1. Extract penultimate-layer features from each model on the test set.
  2. Run t-SNE on each set of features independently.
  3. Produce a 1×3 subplot comparison: Teacher | Baseline | Distilled.
  4. Save the figure and optionally log it to TensorBoard.

Usage::

    python -m src.utils.visualize \\
        --teacher_ckpt ./checkpoints/teacher/best_model.pth \\
        --baseline_ckpt ./checkpoints/baseline/best_model.pth \\
        --distilled_ckpt ./checkpoints/distilled/best_model.pth \\
        --data_dir ./data/hmdb51 \\
        --output_path ./figures/tsne_comparison.png

Programmatic usage::

    from src.utils.visualize import extract_penultimate_features, plot_tsne

    features, labels = extract_penultimate_features(model, dataloader, device)
    fig = plot_tsne(
        {"Teacher": (t_feats, t_labels), "Student": (s_feats, s_labels)},
        class_names=dataset.classes,
    )
    fig.savefig("tsne.png")
"""

import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ======================================================================
# FEATURE EXTRACTION
# ======================================================================

@torch.no_grad()
def extract_penultimate_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features from a model.

    Hooks into the global average pooling layer to capture the feature
    vector before the final classifier head.

    Args:
        model: Trained model (Teacher or Student).
        dataloader: Test data loader.
        device: Computation device.
        max_samples: Maximum number of samples to extract (for t-SNE
            performance; 2000 is a good default).

    Returns:
        Tuple of:
          - ``features``: Array of shape ``(N, D)`` — feature vectors.
          - ``labels``: Array of shape ``(N,)`` — class indices.
    """
    model.eval()
    model.to(device)

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []
    total_collected = 0

    # Register hook on avgpool
    hook_output = {}

    def hook_fn(module, input, output):
        hook_output["features"] = output

    # Find and hook the avgpool layer
    if hasattr(model, "avgpool"):
        hook = model.avgpool.register_forward_hook(hook_fn)
    else:
        raise AttributeError(
            f"Model {type(model).__name__} has no 'avgpool' attribute. "
            f"Cannot extract penultimate features."
        )

    for clips, batch_labels in dataloader:
        if total_collected >= max_samples:
            break

        clips = clips.to(device)
        _ = model(clips)

        # Get hooked features: (B, D, 1, 1, 1) → (B, D)
        feat = hook_output["features"]
        feat = feat.view(feat.size(0), -1).cpu().numpy()

        batch_size = min(feat.shape[0], max_samples - total_collected)
        features_list.append(feat[:batch_size])
        labels_list.extend(batch_labels[:batch_size].tolist())
        total_collected += batch_size

    hook.remove()

    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)

    logger.info(
        f"Extracted {features.shape[0]} feature vectors "
        f"(dim={features.shape[1]}) from {type(model).__name__}"
    )

    return features, labels


# ======================================================================
# t-SNE COMPUTATION
# ======================================================================

def compute_tsne(
    features: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Run t-SNE dimensionality reduction.

    Args:
        features: Input features of shape ``(N, D)``.
        perplexity: t-SNE perplexity parameter. Controls the balance
            between local and global structure. Typical range: [5, 50].
        n_iter: Number of optimization iterations.
        random_state: Random seed for reproducibility.

    Returns:
        2D embedding of shape ``(N, 2)``.
    """
    from sklearn.manifold import TSNE

    logger.info(
        f"Running t-SNE on {features.shape[0]} samples "
        f"(D={features.shape[1]}, perplexity={perplexity})..."
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(features)

    logger.info(f"t-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}")
    return embedding


# ======================================================================
# PLOTTING
# ======================================================================

def plot_tsne(
    model_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    class_names: Optional[List[str]] = None,
    max_classes_in_legend: int = 15,
    figsize: Tuple[int, int] = (20, 6),
    point_size: int = 8,
    alpha: float = 0.6,
    title: str = "t-SNE Latent Space Comparison",
) -> "matplotlib.figure.Figure":
    """Create a side-by-side t-SNE comparison plot.

    Each model gets its own subplot with color-coded class clusters.

    Args:
        model_data: Dict mapping model names to ``(features, labels)``
            tuples. Features should be raw (not yet t-SNE'd) or
            pre-computed 2D embeddings.
        class_names: Optional list of class names for the legend.
        max_classes_in_legend: Maximum classes to show in legend
            (HMDB-51 has 51 classes; showing all is cluttered).
        figsize: Figure size ``(width, height)`` in inches.
        point_size: Scatter plot marker size.
        alpha: Marker transparency.
        title: Figure suptitle.

    Returns:
        Matplotlib ``Figure`` object.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    n_models = len(model_data)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    # Determine number of unique classes
    all_labels = np.concatenate([
        labels for _, labels in model_data.values()
    ])
    unique_classes = np.unique(all_labels)
    n_classes = len(unique_classes)

    # Color map
    cmap = cm.get_cmap("tab20", min(n_classes, 20))
    colors = [cmap(i % 20) for i in range(n_classes)]

    for ax, (model_name, (features, labels)) in zip(axes, model_data.items()):
        # Run t-SNE if features are high-dimensional
        if features.shape[1] > 2:
            embedding = compute_tsne(features)
        else:
            embedding = features

        # Scatter plot with class colors
        for class_idx in unique_classes:
            mask = labels == class_idx
            label_text = None
            if class_names and class_idx < len(class_names):
                if class_idx < max_classes_in_legend:
                    label_text = class_names[class_idx]

            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[class_idx % len(colors)]],
                s=point_size,
                alpha=alpha,
                label=label_text,
                edgecolors="none",
            )

        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE dim 1", fontsize=10)
        ax.set_ylabel("t-SNE dim 2", fontsize=10)

        if class_names and n_classes <= max_classes_in_legend:
            ax.legend(
                fontsize=6, loc="best", markerscale=2,
                ncol=2, framealpha=0.7,
            )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ======================================================================
# MAIN CLI
# ======================================================================

def main() -> None:
    """Command-line interface for t-SNE visualization."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for cluster/headless use
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="t-SNE latent space visualization for KD comparison"
    )
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--distilled_ckpt", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data/hmdb51")
    parser.add_argument("--annotation_dir", type=str, default="./data/hmdb51_splits")
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=51)
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--output_path", type=str, default="./figures/tsne_comparison.png")
    parser.add_argument("--log_dir", type=str, default="./runs/tsne")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from src.utils.seed import set_seed
    from src.models import build_model
    from src.datasets import build_dataset
    from src.evaluation.evaluate import load_model_from_checkpoint

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build test dataset
    _, _, test_ds = build_dataset(
        dataset_type="video",
        data_dir=args.data_dir,
        annotation_dir=args.annotation_dir,
        split=args.split,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    class_names = getattr(test_ds, "classes", None)

    # Extract features from each available model
    model_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if args.teacher_ckpt:
        logger.info("Extracting Teacher features...")
        teacher = load_model_from_checkpoint(
            "teacher", args.teacher_ckpt, args.num_classes, device=device
        )
        feats, labs = extract_penultimate_features(
            teacher, test_loader, device, args.max_samples
        )
        model_data["Teacher (ResNet3D-50)"] = (feats, labs)
        del teacher

    if args.baseline_ckpt:
        logger.info("Extracting Baseline Student features...")
        baseline = load_model_from_checkpoint(
            "student", args.baseline_ckpt, args.num_classes,
            args.width_mult, device,
        )
        feats, labs = extract_penultimate_features(
            baseline, test_loader, device, args.max_samples
        )
        model_data["Student Baseline"] = (feats, labs)
        del baseline

    if args.distilled_ckpt:
        logger.info("Extracting Distilled Student features...")
        distilled = load_model_from_checkpoint(
            "student", args.distilled_ckpt, args.num_classes,
            args.width_mult, device,
        )
        feats, labs = extract_penultimate_features(
            distilled, test_loader, device, args.max_samples
        )
        model_data["Student Distilled"] = (feats, labs)
        del distilled

    if not model_data:
        logger.error("No model checkpoints provided. Exiting.")
        return

    # Generate t-SNE plot
    fig = plot_tsne(model_data, class_names=class_names)

    # Save figure
    from pathlib import Path
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=150, bbox_inches="tight")
    logger.info(f"t-SNE plot saved → {args.output_path}")

    # Log to TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
        writer.add_figure("t-SNE/comparison", fig, 0)
        writer.close()
        logger.info(f"t-SNE plot logged to TensorBoard → {args.log_dir}")
    except Exception as e:
        logger.warning(f"TensorBoard logging failed: {e}")

    plt.close(fig)


if __name__ == "__main__":
    main()
