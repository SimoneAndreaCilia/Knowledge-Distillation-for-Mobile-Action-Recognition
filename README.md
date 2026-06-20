# Knowledge Distillation for Mobile Action Recognition

## 👥 Group and Project Information
- **Group ID**: [G39]
- **Project ID**: [32]

## 📝 Project Description
This repository addresses the high computational cost of video action recognition on mobile devices by compressing a heavy **3D ResNet-50** teacher model into an ultra-lightweight **3D MobileNet** student via **Knowledge Distillation (KD)** on the HMDB-51 dataset. By utilizing logit-matching, temperature ablations, and feature-based attention transfer, the project successfully bridges the student's performance gap while reducing parameter counts by 5–10x. Comprehensive evaluations track the accuracy trade-offs, model size reductions, and inference latency improvements, backed by t-SNE latent space visualizations.


## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

**Prerequisites:**
You can set up the required environment and install all dependencies using Anaconda:

```bash
git clone https://github.com/SimoneAndreaCilia/Knowledge-Distillation-for-Mobile-Action-Recognition.git
cd Knowledge-Distillation-for-Mobile-Action-Recognition
conda env create -f environment.yml
conda activate dl-project
```

**Dataset:**
Download the HMDB51 dataset videos from [Hugging Face](https://huggingface.co/datasets/jili5044/hmdb51). Extract the 51 action class folders directly into `data/hmdb51/` and the train/test splits into `data/hmdb51_splits/`.

### 2. Network Training
To reproduce the training runs, use the provided configuration files inside `experiments/configs/`.

**Student Baseline Training:**
```bash
python src/training/train_baseline.py --config experiments/configs/student_baseline.yaml
```

**Improved Model Training (Knowledge Distillation + Attention Transfer):**
```bash
python src/training/train_distill.py --config experiments/configs/distill_AT.yaml
```

### 3. Evaluation
To reproduce the evaluation metrics, profiling, and generate the latent space visualizations (t-SNE), run the provided evaluation scripts.

```bash
# Generate t-SNE latent space comparison plots
python src/evaluation/evaluate_tsne.py

# Run comprehensive model comparison (Teacher vs Baselines vs Distilled)
python src/evaluation/comparison.py
```

## 📊 Latent Space Analysis (t-SNE)

To intuitively understand the effectiveness of our Knowledge Distillation (KD) and Attention Transfer (AT) approaches, we visualize the feature representations (latent space) of the different models using t-SNE. The visualizations project the high-dimensional features of 7 action classes from the HMDB-51 dataset into a 2D space.

### Standard t-SNE Scatter Plot

![t-SNE Latent Space Comparison](figures/tsne_comparison0.png)

This plot shows the raw 2D t-SNE projections for the four main training settings:
1. **Teacher (ResNet3D-50)**: The high-capacity teacher model naturally forms well-separated, distinct clusters for different action classes, demonstrating strong feature discrimination.
2. **Student Baseline**: Trained from scratch, the lightweight student model fails to separate the classes effectively. The data points are heavily mixed and scattered, indicating poor generalization and weak feature representation.
3. **Student Distilled (KD)**: By learning from the teacher's soft targets via standard Knowledge Distillation, the student model begins to form more recognizable clusters. The separation is visibly better than the baseline, though some classes still overlap.
4. **Student Distilled + AT**: Combining KD with Attention Transfer yields the best student representation. The clusters become tighter and better separated, more closely mimicking the structural topology of the teacher's latent space.

### Cluster Overlap Analysis (Pairwise IoU)

![t-SNE with Ellipses and IoU](figures/tsne_comparison3.png)

To quantitatively evaluate the visual separation, we fit covariance ellipses to each class cluster and calculate the average pairwise Intersection over Union (IoU). A lower IoU indicates less overlap and better class separation.
- **Teacher**: Achieves a low **ø pairwise IoU of 0.12**, confirming highly distinct and separated class boundaries.
- **Student Baseline**: Shows a high **ø pairwise IoU of 0.32**, numerically validating the severe cluster overlap and confusion observed in the standard plot.
- **Student Distilled (KD)**: Reduces the overlap to an **ø pairwise IoU of 0.23**, showing a clear quantitative improvement in feature learning over the baseline.
- **Student Distilled + AT**: Achieves an **ø pairwise IoU of 0.19**, the lowest overlap among the student models. This confirms that Attention Transfer significantly helps the lightweight student architecture learn a more discriminative, teacher-like feature space, reducing class confusion.
