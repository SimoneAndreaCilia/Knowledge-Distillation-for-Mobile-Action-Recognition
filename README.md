# Knowledge Distillation for Mobile Action Recognition

## 👥 Group and Project Information
- **Group ID**: [G39]
- **Project ID**: [32]

## 📝 Project Description
This repository addresses the high computational cost of video action recognition on mobile devices by compressing a heavy **3D ResNet-50** teacher model into an ultra-lightweight **3D MobileNet** student via **Knowledge Distillation (KD)** on the HMDB-51 dataset. By utilizing logit-matching, temperature ablations, and feature-based attention transfer, the project successfully bridges the student's performance gap while reducing parameter counts by 5–10x. Comprehensive evaluations track the accuracy trade-offs, model size reductions, and inference latency improvements, backed by t-SNE latent space visualizations.


## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

**Prerequisites:**
Explain how the reader can install the environment to run your code.

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
conda env create -f environment.yml
conda activate dl-project
```

**Dataset:**
Explain in 2 lines where to download the data from and in which folder it needs to reside (e.g., `data/raw/`).

### 2. Network Training
Provide the **exact commands** to start the training.

**Baseline Training:**
```bash
python -m src.training.train --config experiments/configs/baseline.yaml
```

**Improved Model Training:**
```bash
python -m src.training.train --config experiments/configs/model_v1.yaml
```

### 3. Evaluation
Provide the commands to reproduce the numbers in your summary table.

```bash
python -m src.evaluation.evaluate --config experiments/configs/model_v1.yaml
```
