# Teacher Model Training Log - Baseline

This document summarizes the training execution for the teacher model used in the Knowledge Distillation for Mobile Action Recognition project.

## Experiment Setup

* **Model Configuration:** A ResNet3D-50 architecture was used as the teacher model. It was initialized with Kinetics-400 pretrained weights. Because the target dataset has 51 classes instead of 400, the fully connected (fc) classification head was randomly initialized.
* **Dataset:** HMDB51 (Split 1).
* **Data Properties:** The dataset was processed into 16-frame clips at a 112x112 resolution, yielding 3,570 training samples and 1,530 validation samples. The training batch size was set to 24.
* **Hardware:** The training was accelerated using an NVIDIA L40S GPU.

## Training Strategy

The model was trained over 40 epochs (0 to 39) using a two-phase transfer learning approach:

* **Phase 1 (Epochs 0-4) - Classifier Warm-up:** The ResNet3D-50 backbone was frozen, meaning only the 104,499 parameters of the new classification head were updated. This prevents early large gradients from distorting the pretrained backbone.
* **Phase 2 (Epochs 5-39) - Full Fine-tuning:** The backbone was completely unfrozen (all 46.3M parameters became trainable). The optimization switched to a differential learning rate strategy, where the backbone learned at a significantly lower rate than the newly initialized head.

## Performance & Results

* **Best Validation Accuracy:** 62.94%.
* **Training Accuracy:** Reached ~88.2% by the final epoch.
* **Execution Time:** The entire training session completed efficiently in roughly 39.3 minutes (0.65 hours).
