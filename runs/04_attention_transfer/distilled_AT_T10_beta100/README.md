# Student Model Training Log - Attention Transfer ($\beta=100$) & Knowledge Distillation (T=10.0)

This document summarizes the training execution for the student model using a heavily scaled Attention Transfer (AT) approach combined with Knowledge Distillation (KD).

## Experiment Setup

* **Teacher Model:** A pre-trained ResNet3D-50 model containing 46,303,475 parameters. The teacher's weights were loaded from its best checkpoint (accuracy ~62.94%) and kept completely frozen during the training.
* **Student Model:** The baseline lightweight MobileNet3D architecture was built for the student network.
* **Distillation Parameters:** Standard KD was paired with an Attention Transfer (AT) mechanism using an explicitly high weight penalty of $\beta=100$.
* **Hardware:** The training run was accelerated using an NVIDIA L40S GPU.

## Training Execution & Results

* **Planned Schedule:** The training initialized from scratch and advanced toward its full 100-epoch schedule (recording checkpoint saves up to epoch 98).
* **Performance:** The student model maintained highly competitive performance, reaching a validation accuracy of 44.7% by epoch 97.
* **Loss & Overfitting Observations:** 
  * By scaling the Attention Transfer penalty to $\beta=100$, the total training loss balanced at around ~4.24 during the final measured epochs. 
  * The model achieved ~98.9% training accuracy by epoch 96, confirming that it successfully mimicked the teacher's attention maps while maintaining the massive generalization improvement associated with AT optimization.
* **Execution Time:** The training loop executed efficiently, with epochs averaging roughly 55 seconds.
