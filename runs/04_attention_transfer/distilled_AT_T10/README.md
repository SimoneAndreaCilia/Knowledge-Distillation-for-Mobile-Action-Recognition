# Student Model Training Log - Attention Transfer & Knowledge Distillation (T=10.0)

This document summarizes the training execution for the student model as part of the Mobile Action Recognition project. This specific run combines standard Knowledge Distillation (KD) with Attention Transfer (AT) to improve the student network's feature representation.

## Experiment Setup

* **Teacher Model:** A pre-trained ResNet3D-50 model containing 46,303,475 parameters. The teacher's weights were loaded from its best checkpoint (accuracy ~62.94%) and kept completely frozen during the training.
* **Student Model:** The baseline lightweight MobileNet3D architecture containing 2,419,379 trainable parameters with a file size of roughly 9.23 MB.
* **Distillation Parameters:** The standard KD optimization was configured with a Temperature (T) of 10.0 and an Alpha (α) weight of 0.3. This was augmented with an Attention Transfer (AT) loss mechanism, scaled by a Beta (β) parameter equal to 1000.
* **Dataset:** HMDB51 (Split 1) was used, processed into 16-frame clips at a 112x112 resolution.
* **Data Properties:** The experiment utilized 3,570 training samples and 1,530 validation samples with a batch size of 64.
* **Hardware:** The training run was accelerated using an NVIDIA L40S GPU.

## Training Execution & Results

* **Planned Schedule:** The training initialized from scratch (no prior checkpoints) and successfully ran for the full scheduled 100 epochs (0 to 99).
* **Breakthrough Performance:** The addition of Attention Transfer yielded a massive performance leap. The student model achieved a peak validation accuracy of 47.19% at epoch 94 (compared to the ~26-29% plateau observed in standard KD runs).
* **Loss & Overfitting Observation:** 
  * The total training loss values logged (climbing to ~30.8) are significantly higher than standard runs. This safely indicates that the unscaled Attention Transfer penalty is dominating the raw loss calculation.
  * While the model still memorized the training set (reaching ~98.1% training accuracy by the final epochs), the AT mechanism successfully forced the student to learn highly generalizable spatial-temporal features, drastically reducing the generalization gap.
* **Execution Time:** The entire training session completed successfully and efficiently in approximately 1.57 hours (94.0 minutes).
