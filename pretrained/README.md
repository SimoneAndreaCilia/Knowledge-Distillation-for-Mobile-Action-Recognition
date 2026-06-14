# Pre-trained Model Weights

Place downloaded pre-trained model checkpoints in this directory.

## Required files

### Teacher (3D ResNet-50)
- **`r3d50_K_200ep.pth`** — Kinetics-400 pre-trained weights (Hara et al.)
  - Download from: https://github.com/kenshohara/3D-ResNets-PyTorch/releases
  - Size: ~170 MB

## Usage

```bash
python -m src.training.train_baseline \
    --config experiments/configs/teacher.yaml \
    --pretrained --pretrained_source kinetics \
    --pretrained_path ./pretrained/r3d50_K_200ep.pth
```
