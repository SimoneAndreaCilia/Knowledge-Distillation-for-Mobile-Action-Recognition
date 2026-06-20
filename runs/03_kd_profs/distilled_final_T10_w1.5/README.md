# Student Model Training Log - Knowledge Distillation (Wider Student, T=10.0)

This document summarizes the final Knowledge Distillation (KD) training execution for the student model as part of the Mobile Action Recognition project. This specific run tested a higher capacity student network using a width multiplier ($\alpha$) of 1.5.

## Experiment Setup

* **Teacher Model:** A pre-trained ResNet3D-50 model containing 46,303,475 parameters. The teacher's weights were loaded from its best checkpoint (accuracy ~62.94%) and kept completely frozen during the training.
* **Student Model:** A scaled-up MobileNet3D architecture. By increasing the width multiplier to $\alpha=1.5$, the parameter count more than doubled compared to baseline runs, reaching 5,231,251 trainable parameters with a file size of 19.96 MB.
* **Distillation Parameters:** The KD optimization was configured with a Temperature (T) of 10.0 and an Alpha ($\alpha$) weight of 0.3.
* **Dataset:** HMDB51 (Split 1) was used, processed into 16-frame clips at a 112x112 resolution.
* **Data Properties:** The experiment utilized 3,570 training samples and 1,530 validation samples. To accommodate the larger student model within memory limits, the batch size was reduced to 48 (down from 64 in previous runs).
* **Hardware:** The training run was accelerated using an NVIDIA L40S GPU.

## Training Execution & Results

* **Planned Schedule:** The training initialized from scratch and successfully ran for the full scheduled 100 epochs (0 to 99).
* **Best Performance:** The wider student model achieved a peak validation accuracy of 29.08% at epoch 71.
* **Overfitting Observation:** Despite the increased capacity, the model still exhibited severe overfitting. By epoch 99, the training accuracy reached near-perfect levels (99.7%), while validation accuracy stagnated around 28.3%.
* **Execution Time:** The entire training session completed successfully in approximately 1.56 hours (93.7 minutes).
