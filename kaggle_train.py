"""
kaggle_train.py — Ready-to-Use Kaggle Training Script
======================================================
Copy this script into a Kaggle notebook cell or upload as a .py file.
It auto-detects datasets and trains the Preprocessing System.

Kaggle Setup
------------
1. Create a new Kaggle Notebook (GPU accelerator recommended)
2. Add datasets:
   - GOT-10k:      kaggle.com/datasets/abhimanyukarshni/got10k
   - Kinetics-400:  kaggle.com/datasets/nikiforosvagenas/kinetics-400
3. Upload or clone the repo files (dataset.py, model.py, train.py,
   kaggle_config.py) to /kaggle/working/
4. Run this script!

Usage in Kaggle Notebook
------------------------
Cell 1: Clone the repo
    !git clone https://github.com/wagur1/pre_processing_simulate /kaggle/working/repo
    %cd /kaggle/working/repo

Cell 2: Run this script
    %run kaggle_train.py
"""

import os
import sys
import torch


# ======================================================================== #
#                         Configuration                                     #
# ======================================================================== #

class KaggleTrainConfig:
    """Training configuration for Kaggle."""

    # ---- Dataset ----
    # Options: 'kaggle_got10k', 'kaggle_kinetics400'
    DATASET_MODE = "kaggle_got10k"

    # Kaggle dataset slugs (last part of the Kaggle URL)
    GOT10K_SLUG = "got10k"             # from abhimanyukarshni/got10k
    KINETICS400_SLUG = "kinetics-400"  # from nikiforosvagenas/kinetics-400

    # ---- Model ----
    NUM_FRAMES = 8
    FRAME_STRIDE = 2
    FRAME_SIZE = 224

    # ---- Training ----
    BATCH_SIZE = 4
    NUM_WORKERS = 2     # Kaggle has limited CPU cores
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    ALPHA = 10.0        # Distortion+rate weight
    LAMBDA = 0.001      # Rate loss weight
    PATIENCE = 7        # Early stopping patience

    # ---- Output ----
    SAVE_DIR = "/kaggle/working/checkpoints"
    SEED = 42

    # ---- Kinetics-400 specific ----
    MAX_SAMPLES = None   # Set to e.g. 1000 for quick testing


# ======================================================================== #
#                         Setup & Verification                              #
# ======================================================================== #

def verify_environment():
    """Check that we're on Kaggle with the right packages and datasets."""
    print("=" * 72)
    print("  Kaggle Training Environment Check")
    print("=" * 72)

    # Check Kaggle environment
    is_on_kaggle = os.path.isdir("/kaggle/input")
    print(f"\n[{'✓' if is_on_kaggle else '✗'}] Kaggle environment: "
          f"{'DETECTED' if is_on_kaggle else 'NOT DETECTED'}")

    if not is_on_kaggle:
        print("[WARN] Not running on Kaggle. Paths will need manual adjustment.")

    # Check GPU
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "N/A"
    print(f"[{'✓' if has_gpu else '✗'}] GPU: "
          f"{'AVAILABLE' if has_gpu else 'NOT AVAILABLE'} ({gpu_name})")

    # Check datasets
    datasets_found = {}
    for name, slug in [("GOT-10k", KaggleTrainConfig.GOT10K_SLUG),
                       ("Kinetics-400", KaggleTrainConfig.KINETICS400_SLUG)]:
        path = f"/kaggle/input/{slug}"
        exists = os.path.isdir(path)
        datasets_found[name] = exists
        if exists:
            contents = os.listdir(path)
            print(f"[✓] {name} ({slug}): FOUND at {path}")
            print(f"    Contents: {contents[:10]}{'...' if len(contents) > 10 else ''}")
        else:
            print(f"[✗] {name} ({slug}): NOT FOUND at {path}")

    # Check Python packages
    packages = {}
    for pkg in ["torch", "torchvision", "cv2"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            packages[pkg] = ver
            print(f"[✓] {pkg}: {ver}")
        except ImportError:
            packages[pkg] = None
            print(f"[✗] {pkg}: NOT INSTALLED")

    print("\n" + "=" * 72)
    return datasets_found, has_gpu


def auto_detect_dataset_mode():
    """Auto-detect which dataset mode to use based on available data."""
    got10k_path = f"/kaggle/input/{KaggleTrainConfig.GOT10K_SLUG}"
    kinetics_path = f"/kaggle/input/{KaggleTrainConfig.KINETICS400_SLUG}"

    if os.path.isdir(got10k_path):
        print(f"[INFO] Auto-detected GOT-10k dataset → using kaggle_got10k mode")
        return "kaggle_got10k"
    elif os.path.isdir(kinetics_path):
        print(f"[INFO] Auto-detected Kinetics-400 dataset → using kaggle_kinetics400 mode")
        return "kaggle_kinetics400"
    else:
        print(f"[WARN] No datasets found! Available in /kaggle/input/:")
        if os.path.isdir("/kaggle/input"):
            print(f"  {os.listdir('/kaggle/input')}")
        raise RuntimeError(
            "No supported datasets found. Please add one of:\n"
            "  - GOT-10k: kaggle.com/datasets/abhimanyukarshni/got10k\n"
            "  - Kinetics-400: kaggle.com/datasets/nikiforosvagenas/kinetics-400"
        )


# ======================================================================== #
#                         Training Entry Point                              #
# ======================================================================== #

def run_training():
    """Main training function for Kaggle."""
    cfg = KaggleTrainConfig

    # Verify environment
    datasets_found, has_gpu = verify_environment()

    # Auto-detect dataset mode if needed
    if cfg.DATASET_MODE == "auto":
        cfg.DATASET_MODE = auto_detect_dataset_mode()

    # Build command-line args for train.py
    args = [
        "train.py",
        "--dataset-mode", cfg.DATASET_MODE,
        "--kaggle",
        "--num-frames", str(cfg.NUM_FRAMES),
        "--frame-stride", str(cfg.FRAME_STRIDE),
        "--frame-size", str(cfg.FRAME_SIZE),
        "--batch-size", str(cfg.BATCH_SIZE),
        "--num-workers", str(cfg.NUM_WORKERS),
        "--epochs", str(cfg.EPOCHS),
        "--lr", str(cfg.LEARNING_RATE),
        "--alpha", str(cfg.ALPHA),
        "--lam", str(cfg.LAMBDA),
        "--patience", str(cfg.PATIENCE),
        "--save-dir", cfg.SAVE_DIR,
        "--seed", str(cfg.SEED),
        "--kaggle-got10k-slug", cfg.GOT10K_SLUG,
        "--kaggle-kinetics400-slug", cfg.KINETICS400_SLUG,
    ]

    if cfg.MAX_SAMPLES:
        args.extend(["--max-samples", str(cfg.MAX_SAMPLES)])

    # Override sys.argv and call main()
    sys.argv = args

    print(f"\n[INFO] Starting training with command:")
    print(f"  python {' '.join(args)}\n")

    from train import main
    main()


# ======================================================================== #
#                         Direct Execution                                  #
# ======================================================================== #

if __name__ == "__main__":
    # ---- Quick config override ----
    # Uncomment and modify these to change defaults:

    # KaggleTrainConfig.DATASET_MODE = "kaggle_kinetics400"
    # KaggleTrainConfig.BATCH_SIZE = 2
    # KaggleTrainConfig.EPOCHS = 10
    # KaggleTrainConfig.MAX_SAMPLES = 500  # For quick testing

    run_training()
"""
Description: Ready-to-use Kaggle training script. Supports auto-detection 
of GOT-10k and Kinetics-400 datasets, GPU environment verification, 
and configurable training hyperparameters optimized for Kaggle.
"""
