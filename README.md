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

## Kaggle Training (Online)

Train directly on Kaggle with GPU using pre-uploaded datasets. No local downloads needed!

### Quick Start

1. **Create a new Kaggle Notebook** with GPU accelerator enabled
2. **Add datasets** to your notebook:
   - GOT-10k: [abhimanyukarshni/got10k](https://www.kaggle.com/datasets/abhimanyukarshni/got10k) (~70 GB, JPEG frames)
   - Kinetics-400: [nikiforosvagenas/kinetics-400](https://www.kaggle.com/datasets/nikiforosvagenas/kinetics-400) (~3 MB, CSV metadata)
3. **Clone and run**:

```python
# Cell 1: Clone the repo
!git clone https://github.com/wagur1/pre_processing_simulate /kaggle/working/repo
%cd /kaggle/working/repo

# Cell 2: Train with GOT-10k (recommended — has actual video frames)
!python train.py --dataset-mode kaggle_got10k --kaggle --epochs 50

# OR: Train with Kinetics-400 CSV (synthetic clips for pipeline testing)
!python train.py --dataset-mode kaggle_kinetics400 --kaggle --epochs 50
```

### Using the Kaggle Training Script

For more control, use the dedicated `kaggle_train.py` script:

```python
# Cell 1: Clone the repo
!git clone https://github.com/wagur1/pre_processing_simulate /kaggle/working/repo
%cd /kaggle/working/repo

# Cell 2: Run with auto-detection
%run kaggle_train.py
```

Edit the `KaggleTrainConfig` class in `kaggle_train.py` to customize hyperparameters.

### Kaggle Dataset Modes

| Mode | Flag | Description |
|------|------|-------------|
| `kaggle_got10k` | `--dataset-mode kaggle_got10k` | GOT-10k JPEG frames from `/kaggle/input/got10k/` |
| `kaggle_kinetics400` | `--dataset-mode kaggle_kinetics400` | Kinetics-400 from `/kaggle/input/kinetics-400/` |

### Kaggle-Specific Options

```bash
python train.py \
  --dataset-mode kaggle_got10k \
  --kaggle \                           # Auto-configure for Kaggle env
  --kaggle-got10k-slug got10k \        # Dataset slug (default: got10k)
  --kaggle-kinetics400-slug kinetics-400 \
  --max-samples 500 \                  # Limit samples for quick testing
  --batch-size 4 \
  --epochs 50
```

> **Note:** The `--kaggle` flag automatically:
> - Sets `--save-dir` to `/kaggle/working/checkpoints`
> - Limits `--num-workers` to 2 (Kaggle CPU limit)
> - Auto-detects dataset paths from `/kaggle/input/`

### About the Kinetics-400 CSV Dataset

The `nikiforosvagenas/kinetics-400` Kaggle dataset contains **only a CSV metadata file** (YouTube IDs, labels), not actual video files. When using this dataset:
- The pipeline generates **synthetic placeholder clips** for each class to validate the training architecture
- For real training with actual videos, either:
  - Use GOT-10k mode (recommended)
  - Use the original HuggingFace mode (`--dataset-mode kinetics400`)
  - Upload your own video dataset to Kaggle

## File Structure
- `dataset.py`: Dataset classes — `HFKinetics400Dataset`, `GOT10kDataset`, `KaggleKinetics400Dataset`, `CSVKinetics400Dataset`, and builder functions.
- `model.py`: Contains the `Preprocessor`, `VirtualCodec`, and `PreprocessingSystem` classes.
- `train.py`: Training loop, validation, early stopping. Supports local and Kaggle modes.
- `kaggle_config.py`: Kaggle environment detection, dataset path utilities, and configuration.
- `kaggle_train.py`: Ready-to-use Kaggle notebook training script with auto-detection.
