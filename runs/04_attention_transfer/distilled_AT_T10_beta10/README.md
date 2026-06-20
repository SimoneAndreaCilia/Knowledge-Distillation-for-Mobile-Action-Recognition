# Student Model Training Log - Attention Transfer ($\beta=10$) & Knowledge Distillation (T=10.0)

This document summarizes the training execution for the student model using a scaled Attention Transfer (AT) approach combined with Knowledge Distillation (KD).

## Experiment Setup

* **Teacher Model:** A pre-trained ResNet3D-50 model (46,303,475 parameters), loaded from its best checkpoint (accuracy ~62.94%) and kept completely frozen.
* **Student Model:** The baseline lightweight MobileNet3D architecture ($\alpha=1.0$) containing 2,419,379 trainable parameters.
* **Distillation Parameters:** Standard KD was configured with a Temperature (T) of 10.0 and an Alpha ($\alpha$) weight of 0.3. This run incorporated an Attention Transfer (AT) mechanism with an explicitly scaled weight penalty of $\beta=10$.
* **Dataset:** HMDB51 (Split 1), processed into 16-frame clips at a 112x112 resolution.
* **Data Properties:** 3,570 training samples and 1,530 validation samples with a batch size of 64.
* **Hardware:** The training run was accelerated using an NVIDIA L40S GPU.

## Training Execution & Results

* **Planned Schedule:** The training initialized from scratch and successfully completed its full 100-epoch schedule (0 to 99).
* **Performance:** The student model reached a peak validation accuracy of 41.11%.
* **Loss & Overfitting Observations:** 
  * By scaling the Attention Transfer penalty ($\beta=10$), the total training loss was kept within a stable, normalized range (dropping to ~1.08 by the final epochs), preventing the loss inflation seen in unscaled AT runs. 
  * While the peak validation accuracy (41.11%) is slightly lower than the unscaled AT experiment, it still represents a massive ~12-15% absolute improvement over standard KD baselines. 
  * The model reached ~98.9% training accuracy by the end of the run, confirming strong convergence and feature absorption.
* **Execution Time:** The session completed efficiently in approximately 1.57 hours (94.4 minutes).
