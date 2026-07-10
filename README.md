# Preprocessing Framework for Video Machine Vision under Compression

This repository contains the PyTorch training pipeline for the paper **"A Preprocessing Framework for Video Machine Vision under Compression"**.

The framework introduces a trainable preprocessor and a differentiable virtual codec to optimize video compression specifically for downstream machine vision tasks (like action recognition), rather than just for human visual perception.

## Architecture Overview

The system consists of three main components:
1. **Preprocessor (`model.py`)**: A multi-branch CNN that fuses temporal information (inter-frame) and spatial information (intra-frame) to enhance the current frame before compression.
2. **Virtual Codec (`model.py`)**: A differentiable proxy for a video encoder. It utilizes the Straight-Through Estimator (STE) for quantization and includes a Laplacian entropy model to estimate the bit rate (bpp).
3. **Downstream Analyzer (`train.py`)**: A frozen, pre-trained 3D CNN (e.g., R3D-18 / SlowFast) used to calculate the task accuracy loss ($L_{Acc}$).

## Requirements

- Python 3.8+
- PyTorch & Torchvision
- OpenCV (`cv2`)
- HuggingFace `datasets` (if using Kinetics-400 mode)

```bash
pip install torch torchvision opencv-python datasets
```

## Dataset Support

The framework supports two dataset modes:

### 1. GOT-10k (Local Image Sequences)
Ideal for testing tracking robustness. Sequences are organized into pseudo-classes for training and testing.
- **Structure:**
  ```text
  data/
  ├── val/
  │   ├── GOT-10k_Val_000001/
  │   │   ├── 00000001.jpg
  │   │   └── ...
  │   └── list.txt
  └── test/
      ├── GOT-10k_Test_000001/
      └── list.txt
  ```
- **Training Command:**
  ```bash
  python train.py \
    --dataset-mode got10k \
    --train-dir /path/to/got10k/val \
    --test-dir /path/to/got10k/test \
    --batch-size 4 \
    --epochs 50
  ```

### 2. Kinetics-400 (HuggingFace)
Downloads and processes Kinetics-400 directly via the `datasets` library.
- **Training Command:**
  ```bash
  python train.py \
    --dataset-mode kinetics400 \
    --hf-split validation \
    --batch-size 4 \
    --epochs 50
  ```

## Training Details

The framework is optimized jointly using the loss function:
$$L = \alpha \cdot (L_D + \lambda \cdot L_R) + L_{Acc}$$

- **$L_D$ (Distortion Loss)**: MSE between the original and reconstructed frame.
- **$L_R$ (Rate Loss)**: Estimated bits-per-pixel (bpp) from the virtual codec.
- **$L_{Acc}$ (Task Loss)**: Cross-entropy loss from the frozen downstream analyzer.

**Hyperparameters:**
- $\alpha = 10.0$
- $\lambda = 0.001$
- Optimizer: Adam (lr = 1e-4)

The model automatically saves checkpoints (`best_model.pt` and `final_model.pt`) into the `./checkpoints/` directory.

## File Structure
- `dataset.py`: Contains the `VideoFrameDataset`, `HFKinetics400Dataset`, and `GOT10kDataset` classes.
- `model.py`: Contains the `Preprocessor`, `VirtualCodec`, and `PreprocessingSystem` classes.
- `train.py`: Contains the training loop, validation logic, and early stopping mechanism.
