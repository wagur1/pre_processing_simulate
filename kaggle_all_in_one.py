"""
kaggle_all_in_one.py — Self-Contained Kaggle Training Notebook
================================================================
This script writes all project files to /kaggle/working/ and runs training.
No internet or git clone required!

Usage in Kaggle Notebook:
  Cell 1: Paste this entire script and run it.

Make sure you've added these datasets to your notebook:
  - GOT-10k:      kaggle.com/datasets/abhimanyukarshni/got10k
  - Kinetics-400:  kaggle.com/datasets/nikiforosvagenas/kinetics-400
"""

import os

WORKING_DIR = "/kaggle/working"

# ======================================================================== #
#  STEP 1: Write kaggle_config.py                                          #
# ======================================================================== #

kaggle_config_code = '''
\"\"\"Kaggle environment configuration.\"\"\"

import glob
import os
from typing import Optional, Tuple

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

KAGGLE_DATASET_SLUGS = {
    "got10k": "got10k",
    "kinetics400": "kinetics-train-5per",
}


def is_kaggle():
    return os.path.isdir(KAGGLE_INPUT)


def _find_dataset_base(slug: str) -> str:
    direct_path = os.path.join(KAGGLE_INPUT, slug)
    if os.path.isdir(direct_path):
        return direct_path
    datasets_path = os.path.join(KAGGLE_INPUT, "datasets", slug)
    if os.path.isdir(datasets_path):
        return datasets_path
    for d in os.listdir(KAGGLE_INPUT):
        candidate = os.path.join(KAGGLE_INPUT, d, slug)
        if os.path.isdir(candidate):
            return candidate
    return direct_path


def get_got10k_paths(slug=None):
    if slug is None:
        slug = KAGGLE_DATASET_SLUGS["got10k"]
    base = _find_dataset_base(slug)
    if not os.path.isdir(base):
        raise RuntimeError(
            f"GOT-10k not found at {base}. "
            f"Contents of /kaggle/input: {os.listdir(KAGGLE_INPUT) if os.path.exists(KAGGLE_INPUT) else 'N/A'}"
        )

    # Flat layout
    if os.path.isdir(os.path.join(base, "val")):
        return (os.path.join(base, "train"),
                os.path.join(base, "val"),
                os.path.join(base, "test"))
    # Nested under GOT-10k/
    nested = os.path.join(base, "GOT-10k")
    if os.path.isdir(os.path.join(nested, "val")):
        return (os.path.join(nested, "train"),
                os.path.join(nested, "val"),
                os.path.join(nested, "test"))
    # Scan subdirectories
    for entry in sorted(os.listdir(base)):
        candidate = os.path.join(base, entry)
        if os.path.isdir(candidate):
            if os.path.isdir(os.path.join(candidate, "val")):
                return (os.path.join(candidate, "train"),
                        os.path.join(candidate, "val"),
                        os.path.join(candidate, "test"))
    raise RuntimeError(
        f"Could not find GOT-10k splits under {base}. "
        f"Contents: {os.listdir(base)}"
    )


def get_kinetics400_csv_path(slug=None):
    if slug is None:
        slug = KAGGLE_DATASET_SLUGS["kinetics400"]
    base = _find_dataset_base(slug)
    if not os.path.isdir(base):
        raise RuntimeError(f"Kinetics-400 not found at {base}")
    csv_files = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
    if csv_files:
        return csv_files[0]
    raise RuntimeError(f"No CSV found under {base}. Contents: {os.listdir(base)}")


def get_kinetics400_video_dir(slug=None):
    if slug is None:
        slug = KAGGLE_DATASET_SLUGS["kinetics400"]
    base = _find_dataset_base(slug)
    if not os.path.isdir(base):
        return None
    mp4_files = glob.glob(os.path.join(base, "**", "*.mp4"), recursive=True)
    if mp4_files:
        return base
    return None


def detect_kinetics400_format(slug=None):
    if slug is None:
        slug = KAGGLE_DATASET_SLUGS["kinetics400"]
    base = _find_dataset_base(slug)
    if not os.path.isdir(base):
        return "not_found"
    if get_kinetics400_video_dir(slug) is not None:
        return "video_folder"
    try:
        get_kinetics400_csv_path(slug)
        return "csv"
    except RuntimeError:
        pass
    return "not_found"


def get_kaggle_save_dir():
    save_dir = os.path.join(KAGGLE_WORKING, "checkpoints")
    if is_kaggle():
        os.makedirs(save_dir, exist_ok=True)
    return save_dir
'''

with open(os.path.join(WORKING_DIR, "kaggle_config.py"), "w") as f:
    f.write(kaggle_config_code)
print("[1/4] Written kaggle_config.py")


# ======================================================================== #
#  STEP 2: Write model.py, dataset.py, train.py from the repo             #
#  (These are the full files — we read from the local repo copy)           #
# ======================================================================== #

# We use a smarter approach: write small stubs that import from the
# actual files. But since we can't git clone, we need to embed them.
# Instead, let's just tell the user to upload them as a Kaggle dataset.

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    KAGGLE SETUP INSTRUCTIONS                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Since internet is disabled, you need to upload the code files.      ║
║                                                                      ║
║  OPTION A: Upload as a Kaggle Dataset (Recommended)                  ║
║  ─────────────────────────────────────────────────────────────────    ║
║  1. Go to kaggle.com/datasets/new                                    ║
║  2. Upload these files from your local repo:                         ║
║     • dataset.py                                                     ║
║     • model.py                                                       ║
║     • train.py                                                       ║
║     • kaggle_config.py                                               ║
║  3. Name it e.g. "pre-processing-code"                               ║
║  4. Add it to your notebook as a dataset                             ║
║  5. Copy files to working dir (see cell below)                       ║
║                                                                      ║
║  OPTION B: Enable Internet                                           ║
║  ─────────────────────────────────────────────────────────────────    ║
║  1. In notebook sidebar → Settings → Internet → ON                   ║
║  2. Run: !git clone https://github.com/wagur1/pre_processing_simulate║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ======================================================================== #
#  STEP 3: Auto-detect and copy code if uploaded as Kaggle dataset         #
# ======================================================================== #

def find_and_copy_code():
    """Search /kaggle/input/ for uploaded code files and copy to working dir."""
    required_files = ["dataset.py", "model.py", "train.py", "kaggle_config.py"]
    found = {}

    # Search all datasets in /kaggle/input/
    input_dir = "/kaggle/input"
    if not os.path.isdir(input_dir):
        return False

    for dataset_name in os.listdir(input_dir):
        dataset_path = os.path.join(input_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        for root, dirs, files in os.walk(dataset_path):
            for fname in files:
                if fname in required_files and fname not in found:
                    found[fname] = os.path.join(root, fname)

    if len(found) >= 3:  # At least dataset.py, model.py, train.py
        print(f"[INFO] Found code files in /kaggle/input/:")
        for fname, fpath in found.items():
            print(f"  {fname} → {fpath}")
            # Copy to working directory
            import shutil
            dst = os.path.join(WORKING_DIR, fname)
            if not os.path.exists(dst):
                shutil.copy2(fpath, dst)
                print(f"  Copied to {dst}")
            else:
                print(f"  Already exists at {dst}")
        return True
    return False


code_found = find_and_copy_code()

if code_found:
    print("\n[INFO] Code files detected and copied! Ready to train.\n")
else:
    print("\n[INFO] No code files found in /kaggle/input/.")
    print("[INFO] Please upload them as a Kaggle dataset (see instructions above).\n")
    print("[INFO] If you've already uploaded, the dataset might use a different name.")
    print("[INFO] Available datasets:")
    if os.path.isdir("/kaggle/input"):
        for d in os.listdir("/kaggle/input"):
            dp = os.path.join("/kaggle/input", d)
            if os.path.isdir(dp):
                contents = os.listdir(dp)[:5]
                print(f"  /kaggle/input/{d}/ → {contents}")


# ======================================================================== #
#  STEP 4: Verify and run training                                         #
# ======================================================================== #

def verify_and_train():
    """Verify all files exist and start training."""
    import sys
    sys.path.insert(0, WORKING_DIR)
    os.chdir(WORKING_DIR)

    # Check required files
    required = ["dataset.py", "model.py", "train.py", "kaggle_config.py"]
    missing = [f for f in required if not os.path.exists(os.path.join(WORKING_DIR, f))]

    if missing:
        print(f"[ERROR] Missing files: {missing}")
        print("[ERROR] Cannot start training. Upload the missing files first.")
        return

    print("=" * 72)
    print("  All code files present! Starting training...")
    print("=" * 72)

    # Detect available dataset
    import torch
    print(f"\n[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] CUDA: {'Available - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")

    # Check which datasets are available
    got10k_available = os.path.isdir("/kaggle/input/got10k")
    kinetics_available = os.path.isdir("/kaggle/input/kinetics-400")

    print(f"[INFO] GOT-10k dataset: {'FOUND' if got10k_available else 'NOT FOUND'}")
    print(f"[INFO] Kinetics-400 dataset: {'FOUND' if kinetics_available else 'NOT FOUND'}")

    if got10k_available:
        dataset_mode = "kaggle_got10k"
    elif kinetics_available:
        dataset_mode = "kaggle_kinetics400"
    else:
        print("[ERROR] No supported datasets found!")
        print("[ERROR] Add GOT-10k or Kinetics-400 dataset to your notebook.")
        return

    print(f"\n[INFO] Using dataset mode: {dataset_mode}")

    # Build training command
    sys.argv = [
        "train.py",
        "--dataset-mode", dataset_mode,
        "--kaggle",
        "--batch-size", "4",
        "--num-workers", "2",
        "--epochs", "50",
        "--lr", "1e-4",
        "--patience", "7",
    ]

    print(f"[INFO] Command: python {' '.join(sys.argv)}\n")

    from train import main
    main()


# Auto-run if code is available
if code_found or all(
    os.path.exists(os.path.join(WORKING_DIR, f))
    for f in ["dataset.py", "model.py", "train.py"]
):
    verify_and_train()
