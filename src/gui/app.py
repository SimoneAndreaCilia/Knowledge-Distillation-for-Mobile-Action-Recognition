# -*- coding: utf-8 -*-
"""
GUI Demo — Knowledge Distillation for Action Recognition
==========================================================

Interactive Gradio web application to visually demonstrate and compare
the inference performance of Teacher, Student Baseline, and Distilled
models on HMDB-51 action recognition videos.

Features:
    - Single-model inference with Top-5 predictions and confidence bars
    - Side-by-side comparison of all models on the same video
    - HMDB-51 dataset browser (class → video selection)
    - Upload custom videos (.avi, .mp4)
    - Model caching for instant re-inference
    - "Show all variants" toggle for advanced ablation checkpoints

Usage::

    python -m src.gui.app

    # Or from the project root:
    python src/gui/app.py
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.evaluation.inference import preprocess_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "hmdb51"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# HMDB-51 class names (alphabetical order, matching model output indices)
CLASS_NAMES: List[str] = sorted(
    [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
) if DATA_DIR.exists() else []

NUM_CLASSES = len(CLASS_NAMES) if CLASS_NAMES else 51

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
# Each entry stores metadata + loading configuration for a checkpoint.

MODELS_MAIN: Dict[str, Dict[str, Any]] = {
    "Teacher (ResNet3D-50) — 62.94%": {
        "model_name": "teacher",
        "checkpoint": str(CHECKPOINTS_DIR / "teacher" / "best_model.pth"),
        "width_mult": 1.0,
        "params_m": 46.30,
        "accuracy": 62.94,
        "size_mb": 176.6,
        "description": "3D ResNet-50 fine-tuned from Kinetics-400 weights. "
                       "Upper-bound reference for the KD pipeline.",
    },
    "Student Baseline — 20.13%": {
        "model_name": "student",
        "checkpoint": str(CHECKPOINTS_DIR / "student_baseline" / "best_model.pth"),
        "width_mult": 1.0,
        "params_m": 2.42,
        "accuracy": 20.13,
        "size_mb": 9.23,
        "description": "3D MobileNet trained from scratch on HMDB-51. "
                       "Lower-bound baseline without any knowledge transfer.",
    },
    "Student Distilled (KD T=10) — 29.15%": {
        "model_name": "student",
        "checkpoint": str(CHECKPOINTS_DIR / "distilled_T10" / "best_model.pth"),
        "width_mult": 1.0,
        "params_m": 2.42,
        "accuracy": 29.15,
        "size_mb": 9.23,
        "description": "3D MobileNet distilled with logit-matching KD at T=10. "
                       "Soft targets transfer 'dark knowledge' from the Teacher.",
    },
    "Student Distilled + AT — 47.19%": {
        "model_name": "student",
        "checkpoint": str(CHECKPOINTS_DIR / "distilled_AT_T10_seed1234" / "best_model.pth"),
        "width_mult": 1.0,
        "params_m": 2.42,
        "accuracy": 47.19,
        "size_mb": 9.23,
        "description": "3D MobileNet with KD (T=10) + Attention Transfer (β=1000). "
                       "Best student model — bridges 68% of the Teacher-Baseline gap.",
    },
}

MODELS_ADVANCED: Dict[str, Dict[str, Any]] = {
    "Student KD T=1": {
        "model_name": "student",
        "checkpoint": str(CHECKPOINTS_DIR / "distilled_T1" / "best_model.pth"),
        "width_mult": 1.0,
        "params_m": 2.42,
        "accuracy": None,
        "size_mb": 9.23,
        "description": "KD with T=1 (hard targets only, equivalent to label smoothing).",
    },
    "Student KD T=5": {
        "model_name": "student",
        "checkpoint": str(CHECKPOINTS_DIR / "distilled_T5" / "best_model.pth"),
        "width_mult": 1.0,
        "params_m": 2.42,
        "accuracy": None,
        "size_mb": 9.23,
        "description": "KD with T=5 (moderately softened distributions).",
    },
    "Student KD T=20": {
        "model_name": "student",
        "checkpoint": str(CHECKPOINTS_DIR / "distilled_T20" / "best_model.pth"),
        "width_mult": 1.0,
        "params_m": 2.42,
        "accuracy": None,
        "size_mb": 9.23,
        "description": "KD with T=20 (highly softened, near-uniform distributions).",
    },
    "Student KD + AT (seed 42)": {
        "model_name": "student",
        "checkpoint": str(CHECKPOINTS_DIR / "distilled_AT_T10" / "best_model.pth"),
        "width_mult": 1.0,
        "params_m": 2.42,
        "accuracy": None,
        "size_mb": 9.23,
        "description": "KD + AT (β=1000) with seed 42 (original run).",
    },
    "Student KD + AT (β=10)": {
        "model_name": "student",
        "checkpoint": str(CHECKPOINTS_DIR / "distilled_AT_T10_beta10" / "best_model.pth"),
        "width_mult": 1.0,
        "params_m": 2.42,
        "accuracy": 40.00,
        "size_mb": 9.23,
        "description": "KD + AT with reduced β=10 (balanced loss weighting).",
    },
    "Student KD + AT (β=100)": {
        "model_name": "student",
        "checkpoint": str(CHECKPOINTS_DIR / "distilled_AT_T10_beta100" / "best_model.pth"),
        "width_mult": 1.0,
        "params_m": 2.42,
        "accuracy": None,
        "size_mb": 9.23,
        "description": "KD + AT with intermediate β=100 (ablation).",
    },
    "Student KD T=10 (α=1.5)": {
        "model_name": "student",
        "checkpoint": str(CHECKPOINTS_DIR / "distilled_final_T10_w1.5" / "best_model.pth"),
        "width_mult": 1.5,
        "params_m": 4.70,
        "accuracy": None,
        "size_mb": 40.2,
        "description": "Wider student (α=1.5, ~4.7M params) with KD T=10. "
                       "~7× compression vs Teacher.",
    },
}


# ---------------------------------------------------------------------------
# Model Cache
# ---------------------------------------------------------------------------
_model_cache: Dict[str, torch.nn.Module] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_all_models(show_advanced: bool = False) -> Dict[str, Dict[str, Any]]:
    """Return the active model registry based on the advanced toggle."""
    models = dict(MODELS_MAIN)
    if show_advanced:
        models.update(MODELS_ADVANCED)
    return models


def _load_model(model_key: str) -> torch.nn.Module:
    """Load a model with caching to avoid redundant disk I/O.

    On first load, builds the architecture, loads checkpoint weights,
    and moves to the available device (GPU/CPU). Subsequent calls
    return the cached instance immediately.

    Args:
        model_key: Display name key into MODELS_MAIN or MODELS_ADVANCED.

    Returns:
        The model in eval mode on the target device.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        KeyError: If the model_key is not recognized.
    """
    if model_key in _model_cache:
        return _model_cache[model_key]

    all_models = {**MODELS_MAIN, **MODELS_ADVANCED}
    if model_key not in all_models:
        raise KeyError(f"Unknown model: '{model_key}'")

    config = all_models[model_key]
    ckpt_path = config["checkpoint"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint non trovato: {ckpt_path}\n"
            f"Assicurati di aver eseguito il training per questo modello."
        )

    # Build architecture
    model = build_model(
        model_name=config["model_name"],
        num_classes=NUM_CLASSES,
        pretrained=False,
        width_mult=config["width_mult"],
    )

    # Load trained weights
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(_device)
    model.eval()
    _model_cache[model_key] = model
    return model


# ---------------------------------------------------------------------------
# Inference Logic
# ---------------------------------------------------------------------------

def _run_inference(video_path: str, model_key: str) -> Dict[str, float]:
    """Run inference on a single video and return top-5 predictions.

    Args:
        video_path: Path to the video file.
        model_key: Model identifier key.

    Returns:
        Dict mapping class names to confidence percentages (top-5).
    """
    model = _load_model(model_key)
    clip_tensor = preprocess_video(video_path).to(_device)

    with torch.no_grad():
        logits = model(clip_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)

    results = {}
    for i in range(5):
        idx = top5_indices[0][i].item()
        prob = top5_probs[0][i].item()
        if idx < len(CLASS_NAMES):
            label = CLASS_NAMES[idx]
        else:
            label = f"Class {idx}"
        results[label] = round(prob, 4)

    return results


# ---------------------------------------------------------------------------
# Dataset Browser Helpers
# ---------------------------------------------------------------------------

def _get_class_list() -> List[str]:
    """Return sorted list of HMDB-51 action classes."""
    return CLASS_NAMES


def _get_videos_for_class(class_name: str) -> List[str]:
    """Return list of video filenames for a given action class."""
    if not class_name:
        return []
    class_dir = DATA_DIR / class_name
    if not class_dir.exists():
        return []
    videos = sorted([
        f.name for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".avi", ".mp4", ".mkv", ".mov")
    ])
    return videos


def _get_video_path(class_name: str, video_name: str) -> Optional[str]:
    """Resolve the full path for a dataset video."""
    if not class_name or not video_name:
        return None
    path = DATA_DIR / class_name / video_name
    return str(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Gradio Callback Functions
# ---------------------------------------------------------------------------

def classify_single(
    model_key: str,
    uploaded_video: Optional[str],
    dataset_class: Optional[str],
    dataset_video: Optional[str],
    video_source: str,
    progress=None,
) -> Tuple[Dict[str, float], str, str]:
    """Classify a video with a single selected model.

    Returns:
        Tuple of (top5_confidences, model_info_text, status_message).
    """
    import gradio as gr

    # Determine video path
    if video_source == "📂 Dataset HMDB-51":
        video_path = _get_video_path(dataset_class, dataset_video)
        if not video_path:
            raise gr.Error("Seleziona una classe e un video dal dataset.")
        ground_truth = dataset_class
    else:
        if not uploaded_video:
            raise gr.Error("Carica un file video prima di classificare.")
        video_path = uploaded_video
        ground_truth = None

    if not model_key:
        raise gr.Error("Seleziona un modello.")

    # Get model config
    all_models = {**MODELS_MAIN, **MODELS_ADVANCED}
    config = all_models.get(model_key, {})

    # Status: loading
    gr.Info(f"⏳ Caricamento modello: {model_key}...")

    t0 = time.time()
    try:
        results = _run_inference(video_path, model_key)
    except FileNotFoundError as e:
        raise gr.Error(str(e))
    elapsed = time.time() - t0

    # Build model info
    info_lines = []
    if config.get("description"):
        info_lines.append(f"📋 {config['description']}")
    info_lines.append(f"⚙️ Parametri: {config.get('params_m', '?')}M")
    info_lines.append(f"💾 Dimensione: {config.get('size_mb', '?')} MB")
    if config.get("accuracy"):
        info_lines.append(f"📊 Accuracy test set: {config['accuracy']}%")
    info_lines.append(f"⏱️ Tempo inferenza: {elapsed:.2f}s")
    model_info = "\n".join(info_lines)

    # Status
    top1_class = list(results.keys())[0]
    top1_conf = list(results.values())[0] * 100
    status = f"✅ Predizione: **{top1_class}** ({top1_conf:.1f}%)"
    if ground_truth:
        if top1_class == ground_truth:
            status += f"  ·  🎯 Corretto! (ground truth: {ground_truth})"
        else:
            status += f"  ·  ❌ Errato (ground truth: {ground_truth})"

    return results, model_info, status


def compare_all_models(
    uploaded_video: Optional[str],
    dataset_class: Optional[str],
    dataset_video: Optional[str],
    video_source: str,
    show_advanced: bool,
    progress=None,
) -> Tuple[Any, str]:
    """Run inference with all selected models and return comparison results.

    Returns:
        Tuple of (plotly_figure, summary_markdown).
    """
    import gradio as gr

    # Determine video path
    if video_source == "📂 Dataset HMDB-51":
        video_path = _get_video_path(dataset_class, dataset_video)
        if not video_path:
            raise gr.Error("Seleziona una classe e un video dal dataset.")
        ground_truth = dataset_class
    else:
        if not uploaded_video:
            raise gr.Error("Carica un file video prima di confrontare.")
        video_path = uploaded_video
        ground_truth = None

    models = _get_all_models(show_advanced)
    all_results = {}
    total = len(models)

    gr.Info(f"⏳ Confronto di {total} modelli in corso...")

    for i, (key, config) in enumerate(models.items()):
        ckpt_path = config["checkpoint"]
        if not os.path.exists(ckpt_path):
            all_results[key] = {"error": "Checkpoint mancante"}
            continue
        try:
            results = _run_inference(video_path, key)
            all_results[key] = results
        except Exception as e:
            all_results[key] = {"error": str(e)}

    # Build comparison plot using matplotlib
    fig = _build_comparison_chart(all_results, ground_truth)

    # Build summary markdown
    summary_lines = ["## 📊 Risultati Confronto\n"]
    if ground_truth:
        summary_lines.append(f"**Ground Truth:** `{ground_truth}`\n")

    for model_name, preds in all_results.items():
        if isinstance(preds, dict) and "error" in preds:
            summary_lines.append(f"### ❌ {model_name}\n{preds['error']}\n")
            continue

        top1_class = list(preds.keys())[0]
        top1_conf = list(preds.values())[0] * 100

        if ground_truth:
            icon = "🎯" if top1_class == ground_truth else "❌"
        else:
            icon = "🔍"

        summary_lines.append(
            f"### {icon} {model_name}\n"
            f"**Top-1:** `{top1_class}` ({top1_conf:.1f}%)\n"
        )

        # Show all top-5
        for cls, conf in preds.items():
            bar_len = int(conf * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            summary_lines.append(f"  `{bar}` {cls}: {conf*100:.1f}%")
        summary_lines.append("")

    summary = "\n".join(summary_lines)
    return fig, summary


def _build_comparison_chart(
    all_results: Dict[str, Dict[str, float]],
    ground_truth: Optional[str] = None,
) -> Any:
    """Build a matplotlib comparison bar chart.

    Shows the Top-1 confidence for each model as a horizontal bar chart,
    color-coded by correctness (if ground truth is available).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Filter out errors
    valid = {k: v for k, v in all_results.items()
             if isinstance(v, dict) and "error" not in v}

    if not valid:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Nessun modello disponibile",
                ha="center", va="center", fontsize=16, color="#999")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        return fig

    model_names = list(valid.keys())
    top1_classes = [list(v.keys())[0] for v in valid.values()]
    top1_confs = [list(v.values())[0] * 100 for v in valid.values()]

    # Colors: green if correct, red if wrong, blue if no ground truth
    colors = []
    for cls in top1_classes:
        if ground_truth is None:
            colors.append("#6C63FF")
        elif cls == ground_truth:
            colors.append("#00D68F")
        else:
            colors.append("#FF6B6B")

    # Short names for readability
    short_names = []
    for name in model_names:
        short = name.split("—")[0].strip() if "—" in name else name
        short_names.append(short)

    fig, ax = plt.subplots(figsize=(12, max(4, len(model_names) * 0.9)))

    # Dark theme
    fig.patch.set_facecolor("#16213E")
    ax.set_facecolor("#1a1a2e")

    y_pos = np.arange(len(model_names))
    bars = ax.barh(y_pos, top1_confs, height=0.6, color=colors,
                   edgecolor="white", linewidth=0.5, alpha=0.9)

    # Labels on bars
    for bar, conf, cls in zip(bars, top1_confs, top1_classes):
        label_x = bar.get_width() + 1
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f"{cls}  ({conf:.1f}%)",
                va="center", ha="left", fontsize=10,
                fontweight="bold", color="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=11, color="white", fontweight="bold")
    ax.set_xlabel("Confidenza Top-1 (%)", fontsize=12, color="white", fontweight="bold")
    ax.set_xlim(0, max(top1_confs) * 1.35 if top1_confs else 100)
    ax.invert_yaxis()

    # Style
    ax.tick_params(axis="x", colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.grid(axis="x", alpha=0.15, color="white")

    # Title
    title = "Confronto Top-1 Prediction"
    if ground_truth:
        title += f"  ·  Ground Truth: {ground_truth}"
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=15)

    # Legend
    if ground_truth:
        legend_handles = [
            mpatches.Patch(color="#00D68F", label="✅ Corretto"),
            mpatches.Patch(color="#FF6B6B", label="❌ Errato"),
        ]
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=10, facecolor="#1a1a2e", edgecolor="#444",
                  labelcolor="white")

    plt.tight_layout()
    return fig


def update_model_dropdown(show_advanced: bool) -> Dict:
    """Update the model dropdown choices when advanced toggle changes."""
    import gradio as gr
    models = _get_all_models(show_advanced)
    choices = list(models.keys())
    return gr.update(choices=choices, value=choices[0] if choices else None)


def update_video_dropdown(class_name: str) -> Dict:
    """Update the video dropdown when a class is selected."""
    import gradio as gr
    videos = _get_videos_for_class(class_name)
    return gr.update(choices=videos, value=videos[0] if videos else None)


def update_video_source_visibility(source: str):
    """Toggle visibility between upload and dataset browser."""
    import gradio as gr
    is_dataset = source == "📂 Dataset HMDB-51"
    return (
        gr.update(visible=not is_dataset),  # upload
        gr.update(visible=is_dataset),       # dataset browser
    )


def load_dataset_video_preview(class_name: str, video_name: str) -> Optional[str]:
    """Return the video path for the preview player."""
    return _get_video_path(class_name, video_name)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Global dark styling overrides */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
}

/* Header section */
.header-section {
    text-align: center;
    padding: 20px 0;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.header-section h1 {
    font-size: 2rem !important;
    background: linear-gradient(90deg, #6C63FF, #00D68F);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px !important;
}
.header-section p {
    color: #aaa !important;
    font-size: 1rem;
}

/* Result card styling */
.result-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
}

/* Model info box */
.model-info-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
}

/* Status message styling */
.status-msg .prose {
    font-size: 1.05rem !important;
}

/* Comparison summary */
.comparison-summary .prose {
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
}

/* Video preview container */
.video-preview video {
    border-radius: 12px;
    border: 2px solid rgba(108, 99, 255, 0.3);
}

/* Button styling */
.primary-btn {
    background: linear-gradient(135deg, #6C63FF, #00D68F) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108, 99, 255, 0.35) !important;
}

/* Tab styling */
.tabs .tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
"""


def build_ui():
    """Construct the full Gradio Blocks interface.

    Returns:
        Tuple of (demo, theme, css) so that theme/css can be passed to launch()
        as required by Gradio >= 6.0.
    """
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="#0F0F1A",
        body_background_fill_dark="#0F0F1A",
        block_background_fill="#16213E",
        block_background_fill_dark="#16213E",
        block_border_color="rgba(255, 255, 255, 0.08)",
        block_border_color_dark="rgba(255, 255, 255, 0.08)",
        block_label_text_color="rgba(255, 255, 255, 0.7)",
        block_label_text_color_dark="rgba(255, 255, 255, 0.7)",
        block_title_text_color="white",
        block_title_text_color_dark="white",
        body_text_color="rgba(255, 255, 255, 0.85)",
        body_text_color_dark="rgba(255, 255, 255, 0.85)",
        body_text_color_subdued="rgba(255, 255, 255, 0.5)",
        body_text_color_subdued_dark="rgba(255, 255, 255, 0.5)",
        button_primary_background_fill="linear-gradient(135deg, #6C63FF, #00D68F)",
        button_primary_background_fill_dark="linear-gradient(135deg, #6C63FF, #00D68F)",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        border_color_primary="rgba(108, 99, 255, 0.4)",
        border_color_primary_dark="rgba(108, 99, 255, 0.4)",
        input_background_fill="#1a1a2e",
        input_background_fill_dark="#1a1a2e",
        input_border_color="rgba(255, 255, 255, 0.12)",
        input_border_color_dark="rgba(255, 255, 255, 0.12)",
        shadow_drop="0 4px 20px rgba(0, 0, 0, 0.3)",
        shadow_drop_lg="0 8px 30px rgba(0, 0, 0, 0.4)",
        block_shadow="0 2px 15px rgba(0, 0, 0, 0.25)",
        block_shadow_dark="0 2px 15px rgba(0, 0, 0, 0.25)",
        block_border_width="1px",
        block_border_width_dark="1px",
        block_radius="12px",
        checkbox_background_color="#1a1a2e",
        checkbox_background_color_dark="#1a1a2e",
        checkbox_border_color="rgba(255, 255, 255, 0.2)",
        checkbox_border_color_dark="rgba(255, 255, 255, 0.2)",
    )

    with gr.Blocks(title="KD Action Recognition Demo") as demo:
        # ---- Header ----
        with gr.Column(elem_classes="header-section"):
            gr.Markdown(
                "# 🎬 Knowledge Distillation — Action Recognition\n"
                "Confronta Teacher, Student e modelli Distillati "
                "su video HMDB-51 in tempo reale"
            )

        # ===============================================================
        # TAB 1: SINGLE INFERENCE
        # ===============================================================
        with gr.Tabs(elem_classes="tabs") as tabs:
            with gr.Tab("🔍 Inferenza Singola", id="single"):
                with gr.Row():
                    # ---- Left Column: Video Input ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎥 Video Input")

                        video_source = gr.Radio(
                            label="Sorgente Video",
                            choices=["📤 Upload", "📂 Dataset HMDB-51"],
                            value="📂 Dataset HMDB-51",
                            interactive=True,
                        )

                        # Upload mode
                        with gr.Column(visible=False) as upload_section:
                            uploaded_video = gr.Video(
                                label="Carica un video",
                                sources=["upload"],
                                elem_classes="video-preview",
                            )

                        # Dataset browser mode
                        with gr.Column(visible=True) as dataset_section:
                            dataset_class = gr.Dropdown(
                                label="Classe Azione",
                                choices=_get_class_list(),
                                value=_get_class_list()[0] if _get_class_list() else None,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            dataset_video = gr.Dropdown(
                                label="Video",
                                choices=[],
                                interactive=True,
                                allow_custom_value=False,
                            )
                            video_preview = gr.Video(
                                label="Anteprima",
                                interactive=False,
                                elem_classes="video-preview",
                            )

                    # ---- Right Column: Model Selection + Results ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 🧠 Modello")

                        show_advanced = gr.Checkbox(
                            label="📦 Mostra tutte le varianti (ablation)",
                            value=False,
                            interactive=True,
                        )

                        model_dropdown = gr.Dropdown(
                            label="Seleziona Modello",
                            choices=list(MODELS_MAIN.keys()),
                            value=list(MODELS_MAIN.keys())[0],
                            interactive=True,
                            allow_custom_value=False,
                        )

                        classify_btn = gr.Button(
                            "🔍 Classifica",
                            variant="primary",
                            elem_classes="primary-btn",
                        )

                        # Results
                        gr.Markdown("### 📊 Risultati")

                        status_output = gr.Markdown(
                            value="*In attesa di classificazione...*",
                            elem_classes="status-msg",
                        )

                        results_output = gr.Label(
                            label="Top-5 Predizioni",
                            num_top_classes=5,
                        )

                        model_info = gr.Textbox(
                            label="ℹ️ Informazioni Modello",
                            interactive=False,
                            lines=5,
                            elem_classes="model-info-box",
                        )

                # ---- Event Wiring (Single Inference) ----

                # Toggle video source visibility
                video_source.change(
                    fn=update_video_source_visibility,
                    inputs=[video_source],
                    outputs=[upload_section, dataset_section],
                )

                # Update video list when class changes
                dataset_class.change(
                    fn=update_video_dropdown,
                    inputs=[dataset_class],
                    outputs=[dataset_video],
                )

                # Preview video when selection changes
                dataset_video.change(
                    fn=load_dataset_video_preview,
                    inputs=[dataset_class, dataset_video],
                    outputs=[video_preview],
                )

                # Advanced toggle → update model dropdown
                show_advanced.change(
                    fn=update_model_dropdown,
                    inputs=[show_advanced],
                    outputs=[model_dropdown],
                )

                # Classify button
                classify_btn.click(
                    fn=classify_single,
                    inputs=[
                        model_dropdown,
                        uploaded_video,
                        dataset_class,
                        dataset_video,
                        video_source,
                    ],
                    outputs=[results_output, model_info, status_output],
                )

                # Initialize video list for default class
                demo.load(
                    fn=update_video_dropdown,
                    inputs=[dataset_class],
                    outputs=[dataset_video],
                )

            # ===============================================================
            # TAB 2: MODEL COMPARISON
            # ===============================================================
            with gr.Tab("⚔️ Confronto Modelli", id="compare"):
                gr.Markdown(
                    "### Confronta tutti i modelli sullo stesso video\n"
                    "Esegue l'inferenza con tutti i modelli selezionati e mostra "
                    "i risultati affiancati."
                )

                with gr.Row():
                    # Video input (comparison)
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎥 Video Input")

                        cmp_video_source = gr.Radio(
                            label="Sorgente Video",
                            choices=["📤 Upload", "📂 Dataset HMDB-51"],
                            value="📂 Dataset HMDB-51",
                            interactive=True,
                        )

                        with gr.Column(visible=False) as cmp_upload_section:
                            cmp_uploaded_video = gr.Video(
                                label="Carica un video",
                                sources=["upload"],
                                elem_classes="video-preview",
                            )

                        with gr.Column(visible=True) as cmp_dataset_section:
                            cmp_dataset_class = gr.Dropdown(
                                label="Classe Azione",
                                choices=_get_class_list(),
                                value=_get_class_list()[0] if _get_class_list() else None,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            cmp_dataset_video = gr.Dropdown(
                                label="Video",
                                choices=[],
                                interactive=True,
                                allow_custom_value=False,
                            )
                            cmp_video_preview = gr.Video(
                                label="Anteprima",
                                interactive=False,
                                elem_classes="video-preview",
                            )

                        cmp_show_advanced = gr.Checkbox(
                            label="📦 Includi varianti avanzate",
                            value=False,
                            interactive=True,
                        )

                        compare_btn = gr.Button(
                            "⚔️ Confronta Tutti i Modelli",
                            variant="primary",
                            elem_classes="primary-btn",
                        )

                    # Results (comparison)
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Risultati Confronto")

                        comparison_plot = gr.Plot(
                            label="Confronto Top-1 Confidence",
                        )

                        comparison_summary = gr.Markdown(
                            value="*In attesa del confronto...*",
                            elem_classes="comparison-summary",
                        )

                # ---- Event Wiring (Comparison) ----

                cmp_video_source.change(
                    fn=update_video_source_visibility,
                    inputs=[cmp_video_source],
                    outputs=[cmp_upload_section, cmp_dataset_section],
                )

                cmp_dataset_class.change(
                    fn=update_video_dropdown,
                    inputs=[cmp_dataset_class],
                    outputs=[cmp_dataset_video],
                )

                cmp_dataset_video.change(
                    fn=load_dataset_video_preview,
                    inputs=[cmp_dataset_class, cmp_dataset_video],
                    outputs=[cmp_video_preview],
                )

                compare_btn.click(
                    fn=compare_all_models,
                    inputs=[
                        cmp_uploaded_video,
                        cmp_dataset_class,
                        cmp_dataset_video,
                        cmp_video_source,
                        cmp_show_advanced,
                    ],
                    outputs=[comparison_plot, comparison_summary],
                )

                demo.load(
                    fn=update_video_dropdown,
                    inputs=[cmp_dataset_class],
                    outputs=[cmp_dataset_video],
                )

        # ---- Footer ----
        gr.Markdown(
            "<div style='text-align:center; color:#666; padding:20px 0; "
            "font-size:0.85rem;'>"
            "Knowledge Distillation for Mobile Action Recognition  ·  "
            f"Device: <code>{'CUDA' if torch.cuda.is_available() else 'CPU'}</code>  ·  "
            f"Dataset: HMDB-51 ({len(CLASS_NAMES)} classi)"
            "</div>"
        )

    return demo, theme, CUSTOM_CSS


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    """Launch the Gradio demo server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Device: {_device}")
    logger.info(f"HMDB-51 classes found: {len(CLASS_NAMES)}")
    logger.info(f"Main models: {len(MODELS_MAIN)}")
    logger.info(f"Advanced models: {len(MODELS_ADVANCED)}")

    demo, theme, css = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    main()
