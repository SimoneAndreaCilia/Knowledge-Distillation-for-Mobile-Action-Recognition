# -*- coding: utf-8 -*-
"""
t-SNE Visualization of Latent Space Embeddings
================================================

Generates side-by-side t-SNE scatter plots comparing how four models
(Teacher, Student Baseline, Student Distilled, Student AT Distilled)
cluster the action classes of HMDB-51 in their latent space.

The script:
  1. Loads each model from its checkpoint.
  2. Extracts high-level features (after Global Average Pooling, *before*
     the final linear classifier) for every clip in the test split.
  3. Filters embeddings to a visually manageable subset of classes.
  4. Runs t-SNE and produces a single high-resolution figure with 4 panels.

Usage::

    python -m src.evaluation.evaluate_tsne \
        --teacher_ckpt  checkpoints/teacher/best_model.pth \
        --baseline_ckpt checkpoints/student_baseline/best_model.pth \
        --distilled_ckpt checkpoints/distilled_T10/best_model.pth \
        --distilled_at_ckpt checkpoints/distilled_AT_T10_seed1234/best_model.pth \
        --output figures/tsne_comparison.png

Author : (auto-generated for KD-Action-Recognition project)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets import build_dataset
from src.models import build_model

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_tsne")

# ---------------------------------------------------------------------------
# Default class subset — diverse mix of clearly distinct + potentially
# ambiguous actions for an insightful t-SNE comparison.
# ---------------------------------------------------------------------------
DEFAULT_CLASSES: List[str] = [
    "run",
    "walk",
    "jump",
    "kick_ball",
    "ride_horse",
    "fencing",
    "smile",
]


# ======================================================================
# FEATURE EXTRACTION
# ======================================================================

class _FeatureExtractor:
    """Wraps a model to extract embeddings from the latent space.

    Performs a forward-hook–free approach: manually runs the forward
    pass up to (and including) the global average pooling, skipping the
    final dropout + linear classifier head.

    Attributes:
        model: The loaded ``nn.Module`` (in eval mode, on the target device).
        model_type: ``"teacher"`` or ``"student"`` — controls the forward path.
        device: ``torch.device`` the model lives on.
    """

    def __init__(self, model: nn.Module, model_type: str, device: torch.device) -> None:
        self.model = model
        self.model_type = model_type
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def extract(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract latent embeddings for every sample in *dataloader*.

        Args:
            dataloader: Yields ``(clip, label)`` batches where
                ``clip`` has shape ``(B, 3, T, H, W)``.

        Returns:
            Tuple of ``(embeddings, labels)`` as NumPy arrays.
            ``embeddings`` has shape ``(N, D)`` and ``labels`` shape ``(N,)``.
        """
        all_embeddings: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for batch_idx, (clips, labels) in enumerate(dataloader):
            clips = clips.to(self.device, non_blocking=True)
            emb = self._forward_features(clips)  # (B, D)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(dataloader)} processed "
                    f"({(batch_idx + 1) * dataloader.batch_size} clips)"
                )

        embeddings = np.concatenate(all_embeddings, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        logger.info(f"  ✓ Extraction complete — {embeddings.shape[0]} clips, dim={embeddings.shape[1]}")
        return embeddings, labels_arr

    # ------------------------------------------------------------------
    # Model-specific forward without classification head
    # ------------------------------------------------------------------

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model up to and including GAP, returning the 1-D embedding.

        For **Teacher** (ResNet3D):
            stem → maxpool → layer1…4 → avgpool → flatten  →  (B, 2048)

        For **Student** (MobileNet3D):
            stem → stages → final_conv → avgpool → flatten  →  (B, 1280)
        """
        if self.model_type == "teacher":
            return self._forward_teacher(x)
        else:
            return self._forward_student(x)

    def _forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher (ResNet3D) feature extraction."""
        m = self.model
        x = m.stem(x)
        if m.maxpool is not None:
            x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)          # (B, 2048, 1, 1, 1)
        x = torch.flatten(x, 1)   # (B, 2048)
        return x

    def _forward_student(self, x: torch.Tensor) -> torch.Tensor:
        """Student (MobileNet3D) feature extraction."""
        m = self.model
        x = m.stem(x)
        for stage in m.stages:
            x = stage(x)
        x = m.final_conv(x)
        x = m.avgpool(x)          # (B, 1280, 1, 1, 1)
        x = torch.flatten(x, 1)   # (B, 1280)
        return x


# ======================================================================
# MODEL LOADING
# ======================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    model_type: str,
    num_classes: int = 51,
    width_mult: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Instantiate a model and load weights from a checkpoint.

    Handles checkpoints saved as either a raw ``state_dict`` or a
    dict with a ``"model_state_dict"`` key (the project convention).

    Args:
        checkpoint_path: Path to the ``.pth`` file.
        model_type: ``"teacher"`` or ``"student"``.
        num_classes: Number of output classes.
        width_mult: Width multiplier for the student (ignored for teacher).
        device: Target device.

    Returns:
        The model in eval mode on *device*.
    """
    logger.info(f"Loading {model_type} from: {checkpoint_path}")

    model = build_model(
        model_name=model_type,
        num_classes=num_classes,
        pretrained=False,
        width_mult=width_mult,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  ✓ Loaded ({n_params:.2f}M params) on {device}")
    return model


# ======================================================================
# DATA FILTERING
# ======================================================================

def filter_dataset_by_classes(
    dataset,
    target_classes: List[str],
) -> Tuple[Subset, Dict[int, str]]:
    """Return a ``Subset`` containing only samples from *target_classes*.

    Args:
        dataset: An ``HMDB51VideoDataset`` with ``.classes``, ``.class_to_idx``,
            and ``.samples`` attributes.
        target_classes: List of class name strings (e.g. ``["run", "jump"]``).

    Returns:
        Tuple of ``(subset, idx_to_name)`` where *subset* is a
        ``torch.utils.data.Subset`` and *idx_to_name* maps original
        class indices → human-readable class names.
    """
    # Validate requested classes against the dataset
    available = set(dataset.classes)
    valid_classes: List[str] = []
    for cls in target_classes:
        if cls in available:
            valid_classes.append(cls)
        else:
            logger.warning(f"Class '{cls}' not found in dataset — skipping.")

    if not valid_classes:
        raise ValueError(
            f"None of the requested classes {target_classes} are in the dataset. "
            f"Available: {sorted(available)}"
        )

    target_indices = {dataset.class_to_idx[c] for c in valid_classes}
    idx_to_name = {dataset.class_to_idx[c]: c for c in valid_classes}

    # Find sample indices that belong to the target classes
    keep_indices = [
        i for i, (_, label) in enumerate(dataset.samples) if label in target_indices
    ]
    logger.info(
        f"Filtered dataset: {len(keep_indices)} clips from "
        f"{len(valid_classes)} classes {valid_classes}"
    )

    subset = Subset(dataset, keep_indices)
    return subset, idx_to_name


# ======================================================================
# t-SNE + PLOTTING
# ======================================================================

def run_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Run t-SNE dimensionality reduction on *embeddings*.

    Args:
        embeddings: Array of shape ``(N, D)``.
        perplexity: t-SNE perplexity (controls neighborhood size).
        random_state: Seed for reproducibility.

    Returns:
        2-D projection of shape ``(N, 2)``.
    """
    from sklearn.manifold import TSNE

    n_samples = embeddings.shape[0]
    effective_perplexity = min(perplexity, (n_samples - 1) / 3.0)
    if effective_perplexity < perplexity:
        logger.warning(
            f"Perplexity reduced from {perplexity} to {effective_perplexity:.1f} "
            f"(n_samples={n_samples} is small)."
        )

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        random_state=random_state,
        learning_rate="auto",
        max_iter=1000,
    )
    logger.info(f"Running t-SNE (perplexity={effective_perplexity:.1f}, n={n_samples})…")
    projection = tsne.fit_transform(embeddings)
    logger.info("  ✓ t-SNE complete.")
    return projection


# ======================================================================
# CONFIDENCE ELLIPSE HELPER
# ======================================================================

def _draw_confidence_ellipse(
    ax,
    points: np.ndarray,
    color,
    confidence: float = 0.95,
    fill_alpha: float = 0.12,
    edge_alpha: float = 0.55,
    linewidth: float = 1.5,
) -> None:
    """Draw a confidence ellipse for a 2-D point cloud.

    The ellipse is derived from the eigendecomposition of the sample
    covariance matrix.  Its size is scaled so that it encloses
    ``confidence`` % of the probability mass of the fitted Gaussian
    (Chi-squared quantile with 2 degrees of freedom).

    At 95 % confidence the scale factor is √5.991 ≈ 2.448.

    Args:
        ax: Matplotlib ``Axes`` to draw on.
        points: Array of shape ``(N, 2)`` — the cluster's 2-D coordinates.
        color: Face/edge colour (any matplotlib colour spec).
        confidence: Confidence level for the ellipse (default 0.95).
        fill_alpha: Opacity of the filled interior.
        edge_alpha: Opacity of the ellipse border.
        linewidth: Border line width.
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if len(points) < 3:
        return  # Need at least 3 points for a meaningful covariance

    # Chi-squared scale factor: covers `confidence` % of a 2-D Gaussian
    try:
        from scipy.stats import chi2
        scale = np.sqrt(chi2.ppf(confidence, df=2))
    except ImportError:
        # Hardcoded fallback: sqrt(chi2.ppf(0.95, df=2)) = sqrt(5.991)
        scale = 2.4477

    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)

    # Eigendecomposition: eigenvalues → semi-axes lengths,
    # eigenvectors → rotation of the ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Clamp negative eigenvalues (numerical noise) to zero
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Width / height = 2 * scale * sqrt(eigenvalue)
    width, height = 2.0 * scale * np.sqrt(eigenvalues)

    # Rotation angle from the principal eigenvector (largest eigenvalue last)
    angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))

    # --- Filled interior ---
    ell_fill = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        alpha=fill_alpha,
        linewidth=0,
        zorder=1,
    )
    ax.add_patch(ell_fill)

    # --- Solid border ---
    ell_edge = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        alpha=edge_alpha,
        linewidth=linewidth,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(ell_edge)


def _get_ellipse_params(
    points: np.ndarray,
    confidence: float = 0.95,
) -> Optional[Tuple[np.ndarray, float, float, float]]:
    """Compute ellipse parameters (center, width, height, angle) for a cluster.

    Returns ``None`` if fewer than 3 points are provided.

    Args:
        points: Array of shape ``(N, 2)``.
        confidence: Confidence level.

    Returns:
        Tuple of ``(center, width, height, angle_deg)`` or ``None``.
    """
    if len(points) < 3:
        return None

    try:
        from scipy.stats import chi2
        scale = np.sqrt(chi2.ppf(confidence, df=2))
    except ImportError:
        scale = 2.4477

    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    width, height = 2.0 * scale * np.sqrt(eigenvalues)
    angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    return mean, width, height, angle


def _pairwise_ellipse_iou(
    proj: np.ndarray,
    labels: np.ndarray,
    sorted_class_indices: List[int],
    confidence: float = 0.95,
    n_samples: int = 50_000,
    rng_seed: int = 0,
) -> float:
    """Estimate the mean pairwise IoU of confidence ellipses via Monte Carlo.

    Scatters *n_samples* uniform random points inside the bounding box of
    all ellipses, checks which ellipses contain each point, and accumulates
    pairwise intersection / union counts.

    Args:
        proj: 2-D projections ``(N, 2)``.
        labels: Class labels ``(N,)``.
        sorted_class_indices: Ordered class indices to consider.
        confidence: Confidence level of the ellipses.
        n_samples: Number of Monte Carlo samples.
        rng_seed: Random seed for reproducibility.

    Returns:
        Mean pairwise IoU (float in [0, 1]), or 0.0 if fewer than 2 classes.
    """
    # Gather ellipse parameters per class
    ellipses: List[Tuple[np.ndarray, float, float, float]] = []
    for cls_idx in sorted_class_indices:
        mask = labels == cls_idx
        if not mask.any():
            continue
        params = _get_ellipse_params(proj[mask], confidence)
        if params is not None:
            ellipses.append(params)

    n_ell = len(ellipses)
    if n_ell < 2:
        return 0.0

    # Global bounding box (with margin)
    all_cx = [e[0][0] for e in ellipses]
    all_cy = [e[0][1] for e in ellipses]
    all_r = [max(e[1], e[2]) for e in ellipses]  # max semi-diameter
    x_min = min(cx - r for cx, r in zip(all_cx, all_r))
    x_max = max(cx + r for cx, r in zip(all_cx, all_r))
    y_min = min(cy - r for cy, r in zip(all_cy, all_r))
    y_max = max(cy + r for cy, r in zip(all_cy, all_r))

    rng = np.random.RandomState(rng_seed)
    pts_x = rng.uniform(x_min, x_max, n_samples)
    pts_y = rng.uniform(y_min, y_max, n_samples)

    # For each ellipse, check if each point is inside
    inside = np.zeros((n_ell, n_samples), dtype=bool)
    for i, (center, w, h, ang_deg) in enumerate(ellipses):
        # Transform points into the ellipse's local frame
        ang_rad = np.radians(ang_deg)
        cos_a, sin_a = np.cos(ang_rad), np.sin(ang_rad)
        dx = pts_x - center[0]
        dy = pts_y - center[1]
        local_x = cos_a * dx + sin_a * dy
        local_y = -sin_a * dx + cos_a * dy
        # Standard ellipse equation: (x / a)^2 + (y / b)^2 <= 1
        inside[i] = (local_x / (w / 2.0)) ** 2 + (local_y / (h / 2.0)) ** 2 <= 1.0

    # Pairwise IoU
    iou_sum = 0.0
    n_pairs = 0
    for i in range(n_ell):
        for j in range(i + 1, n_ell):
            inter = np.sum(inside[i] & inside[j])
            union = np.sum(inside[i] | inside[j])
            iou_sum += (inter / union) if union > 0 else 0.0
            n_pairs += 1

    return iou_sum / n_pairs if n_pairs > 0 else 0.0


def plot_tsne_comparison(
    projections: List[np.ndarray],
    labels_list: List[np.ndarray],
    idx_to_name: Dict[int, str],
    titles: List[str],
    output_path: str,
    dpi: int = 300,
    confidence_level: float = 0.95,
    ellipse_fill_alpha: float = 0.12,
) -> None:
    """Create a 1×4 figure of t-SNE scatter plots with a shared legend.

    Each class cluster is annotated with a confidence ellipse derived from
    the eigendecomposition of its 2-D sample covariance matrix.

    Args:
        projections: List of 4 arrays, each ``(N, 2)``.
        labels_list: List of 4 label arrays, each ``(N,)``.
        idx_to_name: Mapping from class index → human-readable name.
        titles: List of 4 subplot titles.
        output_path: File path to save the figure (e.g. ``.png`` or ``.pdf``).
        dpi: Output resolution.
        confidence_level: Confidence level for the ellipses (default 0.95).
        ellipse_fill_alpha: Opacity of the ellipse interior (default 0.12).
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_plots = len(projections)

    # --- Consistent class ordering + HUSL-like palette (pure matplotlib) ---
    sorted_class_indices = sorted(idx_to_name.keys())
    class_names = [idx_to_name[i] for i in sorted_class_indices]
    n_classes = len(class_names)

    # Sample the HSV colormap at evenly spaced hues, then convert to RGB.
    # Slight blend toward white softens saturation for a HUSL-like look.
    cmap = matplotlib.colormaps.get_cmap("hsv")
    raw_colors = [cmap(i / n_classes) for i in range(n_classes)]
    palette = [
        tuple(min(1.0, c * 0.85 + 0.15) for c in rgb[:3])
        for rgb in raw_colors
    ]
    color_map = {name: palette[i] for i, name in enumerate(class_names)}

    # --- Figure setup --------------------------------------------------------
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(6 * n_plots, 5.5),
        dpi=dpi,
        constrained_layout=False,
    )
    if n_plots == 1:
        axes = [axes]

    for ax, proj, labels, title in zip(axes, projections, labels_list, titles):
        # Pass 1: confidence ellipses (drawn first, under the scatter points)
        for cls_idx in sorted_class_indices:
            cls_name = idx_to_name[cls_idx]
            mask = labels == cls_idx
            if not mask.any():
                continue
            _draw_confidence_ellipse(
                ax=ax,
                points=proj[mask],
                color=color_map[cls_name],
                confidence=confidence_level,
                fill_alpha=ellipse_fill_alpha,
            )

        # Pass 2: scatter points on top of the ellipses
        for cls_idx in sorted_class_indices:
            cls_name = idx_to_name[cls_idx]
            mask = labels == cls_idx
            if not mask.any():
                continue
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[color_map[cls_name]],
                s=22,
                alpha=0.80,
                linewidths=0,
                zorder=3,
                label=cls_name,
            )

        # --- Pairwise IoU metric in the subplot title -----------------------
        mean_iou = _pairwise_ellipse_iou(
            proj, labels, sorted_class_indices, confidence=confidence_level,
        )
        ax.set_title(
            f"{title}\n"
            f"\u2300 pairwise IoU: {mean_iou:.2f}",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

        # --- Clip viewport to the bounding box of all ellipses (+ 8% margin) --
        # For a rotated ellipse with semi-axes a = w/2, b = h/2, angle θ:
        #   x_extent = sqrt(a²·cos²θ + b²·sin²θ)
        #   y_extent = sqrt(a²·sin²θ + b²·cos²θ)
        # This is the tight axis-aligned bounding box — no Monte Carlo needed.
        all_x_lo, all_x_hi, all_y_lo, all_y_hi = [], [], [], []
        for cls_idx in sorted_class_indices:
            mask = labels == cls_idx
            if not mask.any():
                continue
            params = _get_ellipse_params(proj[mask], confidence_level)
            if params is None:
                continue
            center, w, h, ang_deg = params
            a, b = w / 2.0, h / 2.0
            theta = np.radians(ang_deg)
            x_ext = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
            y_ext = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
            all_x_lo.append(center[0] - x_ext)
            all_x_hi.append(center[0] + x_ext)
            all_y_lo.append(center[1] - y_ext)
            all_y_hi.append(center[1] + y_ext)

        if all_x_lo:  # at least one valid ellipse
            x_lo, x_hi = min(all_x_lo), max(all_x_hi)
            y_lo, y_hi = min(all_y_lo), max(all_y_hi)
            x_pad = (x_hi - x_lo) * 0.08
            y_pad = (y_hi - y_lo) * 0.08
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Shared legend outside the axes -------------------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[name],
            markersize=8,
            label=name,
        )
        for name in class_names
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(class_names),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "t-SNE — Latent Space Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # Save with tight bounding box to capture the external legend
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Figure saved to: {output_path}")


# ======================================================================
# CLI & MAIN
# ======================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate t-SNE visualizations of the latent space for Teacher, "
            "Student Baseline, Student Distilled, and Student AT Distilled models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Checkpoints ---
    ckpt = parser.add_argument_group("Model Checkpoints")
    ckpt.add_argument(
        "--teacher_ckpt",
        type=str,
        default="checkpoints/teacher/best_model.pth",
        help="Path to the Teacher (ResNet3D-50) checkpoint.",
    )
    ckpt.add_argument(
        "--baseline_ckpt",
        type=str,
        default="checkpoints/student_baseline/best_model.pth",
        help="Path to the Student Baseline checkpoint.",
    )
    ckpt.add_argument(
        "--distilled_ckpt",
        type=str,
        default="checkpoints/distilled_T10/best_model.pth",
        help="Path to the Student Distilled (KD) checkpoint.",
    )
    ckpt.add_argument(
        "--distilled_at_ckpt",
        type=str,
        default="checkpoints/distilled_AT_T10_seed1234/best_model.pth",
        help="Path to the Student Distilled + AT checkpoint.",
    )

    # --- Data ---
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data_dir",
        type=str,
        default="./data/hmdb51",
        help="Path to HMDB-51 video directory.",
    )
    data.add_argument(
        "--annotation_dir",
        type=str,
        default="./data/hmdb51_splits",
        help="Path to HMDB-51 split annotation files.",
    )
    data.add_argument(
        "--split",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="HMDB-51 train/test split number.",
    )
    data.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Subset of class names to visualize.",
    )
    data.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    data.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )

    # --- Model ---
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument(
        "--num_classes",
        type=int,
        default=51,
        help="Number of output classes.",
    )
    model_grp.add_argument(
        "--width_mult",
        type=float,
        default=1.0,
        help="Width multiplier for the Student models.",
    )

    # --- t-SNE ---
    tsne_grp = parser.add_argument_group("t-SNE")
    tsne_grp.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    tsne_grp.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for t-SNE reproducibility.",
    )
    tsne_grp.add_argument(
        "--confidence_level",
        type=float,
        default=0.95,
        help="Confidence level for the ellipses (e.g. 0.95 = 95%% CI).",
    )
    tsne_grp.add_argument(
        "--ellipse_alpha",
        type=float,
        default=0.12,
        help="Opacity (alpha) of the filled ellipse interior.",
    )

    # --- Output ---
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output",
        type=str,
        default="figures/tsne_comparison.png",
        help="Output path for the saved figure (.png or .pdf).",
    )
    out.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point: load models, extract features, run t-SNE, plot."""
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Build the test dataset (full), then create a filtered subset
    # ------------------------------------------------------------------
    logger.info("Building HMDB-51 test dataset…")
    _, test_dataset, _ = build_dataset(
        dataset_type="video",
        data_dir=args.data_dir,
        annotation_dir=args.annotation_dir,
        split=args.split,
        num_frames=16,
        frame_size=112,
    )

    filtered_subset, idx_to_name = filter_dataset_by_classes(
        test_dataset,
        target_classes=args.classes,
    )

    dataloader = DataLoader(
        filtered_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # 2. Define models to evaluate
    # ------------------------------------------------------------------
    model_configs: List[Dict] = [
        {
            "name": "Teacher (ResNet3D-50)",
            "ckpt": args.teacher_ckpt,
            "type": "teacher",
        },
        {
            "name": "Student Baseline",
            "ckpt": args.baseline_ckpt,
            "type": "student",
        },
        {
            "name": "Student Distilled (KD)",
            "ckpt": args.distilled_ckpt,
            "type": "student",
        },
        {
            "name": "Student Distilled + AT",
            "ckpt": args.distilled_at_ckpt,
            "type": "student",
        },
    ]

    # ------------------------------------------------------------------
    # 3. Feature extraction + t-SNE for each model
    # ------------------------------------------------------------------
    all_projections: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_titles: List[str] = []

    for cfg in model_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {cfg['name']}")
        logger.info(f"{'='*60}")

        model = load_model_from_checkpoint(
            checkpoint_path=cfg["ckpt"],
            model_type=cfg["type"],
            num_classes=args.num_classes,
            width_mult=args.width_mult,
            device=device,
        )

        extractor = _FeatureExtractor(model, model_type=cfg["type"], device=device)
        embeddings, labels = extractor.extract(dataloader)

        projection = run_tsne(
            embeddings,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )

        all_projections.append(projection)
        all_labels.append(labels)
        all_titles.append(cfg["name"])

        # Free GPU memory before loading the next model
        del model, extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Plot comparison figure
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("Generating t-SNE comparison figure…")
    logger.info(f"{'='*60}")

    plot_tsne_comparison(
        projections=all_projections,
        labels_list=all_labels,
        idx_to_name=idx_to_name,
        titles=all_titles,
        output_path=args.output,
        dpi=args.dpi,
        confidence_level=args.confidence_level,
        ellipse_fill_alpha=args.ellipse_alpha,
    )

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
