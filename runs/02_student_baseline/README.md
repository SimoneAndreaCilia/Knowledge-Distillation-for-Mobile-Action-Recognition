# Student Model Training Log - Baseline (No Distillation)

This document summarizes the training execution for the student baseline model. This run trained the lightweight student architecture from scratch without any Knowledge Distillation or Attention Transfer mechanisms, serving as the lower-bound baseline for the project.

## Experiment Setup

* **Student Model:** A lightweight MobileNet3D architecture containing 2,419,379 trainable parameters.
* **Distillation Parameters:** None. The model was trained purely on the cross-entropy loss from the ground-truth labels.
* **Dataset:** HMDB51 (Split 1), processed into 16-frame clips at a 112x112 resolution.
* **Data Properties:** The experiment utilized 3,570 training samples and 1,530 validation samples.
* **Hardware:** The training run was accelerated using an NVIDIA L40S GPU.

## Training Execution & Results

* **Planned Schedule:** The training ran up to epoch 85 before stopping.
* **Best Performance:** The student model reached a peak validation accuracy of 20.13% at epoch 62. This highlights the difficulty the small architecture has in learning generalized features from scratch.
* **Overfitting Observation:** The model exhibited extreme overfitting. By epoch 85, the training accuracy climbed to 99.86%, while the validation accuracy degraded and stagnated around 19.08%. This confirms that the model simply memorized the training set without learning generalizable spatial-temporal representations.
