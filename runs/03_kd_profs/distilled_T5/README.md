# Student Model Training Log - Knowledge Distillation (T=5.0)

This document summarizes the Knowledge Distillation (KD) training execution for the student model as part of the Mobile Action Recognition project.

## Experiment Setup

* **Teacher Model:** A pre-trained ResNet3D-50 model with 46,303,475 parameters, which was kept completely frozen during the training.
* **Student Model:** A lightweight MobileNet3D architecture containing 2,419,379 trainable parameters and a file size of roughly 9.23 MB.
* **Distillation Parameters:** The KD optimization was configured with a Temperature (T) of 5.0 and an Alpha (α) weight of 0.3.
* **Dataset:** HMDB51 (Split 1) was used, processed into 16-frame clips at a 112x112 resolution.
* **Data Properties:** The experiment utilized 3,570 training samples and 1,530 validation samples with a batch size of 64.
* **Hardware:** The training run was accelerated using an NVIDIA L40S GPU.

## Training Execution & Interruption

* **Planned Schedule:** The training initialized from scratch (no prior checkpoints) and was scheduled to run for 100 epochs, from epoch 0 to 99.
* **Best Performance:** The student model reached a peak validation accuracy of 28.43% at epoch 47.
* **Overfitting Observation:** In the later epochs, the model exhibited significant overfitting; by epoch 86, the training accuracy exceeded 98.5%, while the validation accuracy stagnated around 27.6%.
* **Stopped Early:** The training script did not finish its planned schedule. It got stopped during the validation phase of Epoch 87 to not waste too much power.
