# Student Model Training Log - Knowledge Distillation (T=20.0)

This document summarizes the Knowledge Distillation (KD) training execution for the student model as part of the Mobile Action Recognition project, testing a highly smoothed probability distribution (T=20.0).

## Experiment Setup

* **Teacher Model:** A pre-trained ResNet3D-50 model containing 46,303,475 parameters. The teacher's weights were loaded from its best checkpoint and kept completely frozen during the training.
* **Student Model:** A lightweight MobileNet3D architecture containing 2,419,379 trainable parameters with a file size of roughly 9.23 MB.
* **Distillation Parameters:** The KD optimization was configured with a high Temperature (T) of 20.0 and an Alpha (α) weight of 0.3.
* **Dataset:** HMDB51 (Split 1) was used, processed into 16-frame clips at a 112x112 resolution.
* **Data Properties:** The experiment utilized 3,570 training samples and 1,530 validation samples with a batch size of 64.
* **Hardware:** The training run was accelerated using an NVIDIA L40S GPU.

## Training Execution & Results

* **Planned Schedule:** The training initialized from scratch (no prior checkpoints) and successfully ran for the full scheduled 100 epochs (0 to 99).
* **Best Performance:** The student model reached a peak validation accuracy of 26.54%.
* **Overfitting Observation:** Similar to previous KD runs, the model exhibited severe overfitting in the final stages; by epoch 99, the training accuracy climbed to 99.1%, while the validation accuracy stagnated around 25.3%.
* **Execution Time:** The entire training session completed successfully in approximately 1.55 hours (92.8 minutes).
