"""Kaggle environment configuration for the pre_processing_simulate project.

Provides utilities for detecting the Kaggle runtime environment,
resolving dataset paths for GOT-10k and Kinetics-400, and managing
checkpoint save directories. All path resolution handles variable
dataset directory structures that may differ across Kaggle dataset
uploads.
"""

import glob
import os
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KAGGLE_INPUT: str = "/kaggle/input"
KAGGLE_WORKING: str = "/kaggle/working"

KAGGLE_DATASET_SLUGS: dict[str, str] = {
    "got10k": "got10k",            # abhimanyukarshni/got10k
    "kinetics400": "kinetics-400",  # nikiforosvagenas/kinetics-400
}


# ---------------------------------------------------------------------------
# Environment detection & path resolution
# ---------------------------------------------------------------------------

def is_kaggle() -> bool:
    """Check whether the code is running inside a Kaggle notebook.

    Returns:
        True if the ``/kaggle/input`` directory exists, False otherwise.
    """
    return os.path.isdir(KAGGLE_INPUT)


def _find_dataset_base(slug: str) -> str:
    """Find the base directory for a dataset slug.
    
    Kaggle sometimes mounts datasets directly at /kaggle/input/{slug},
    but sometimes nests them under /kaggle/input/datasets/{slug} or similar.
    """
    # 1. Check direct mount
    direct_path = os.path.join(KAGGLE_INPUT, slug)
    if os.path.isdir(direct_path):
        return direct_path
        
    # 2. Check under 'datasets'
    datasets_path = os.path.join(KAGGLE_INPUT, "datasets", slug)
    if os.path.isdir(datasets_path):
        return datasets_path
        
    # 3. Search one level deep
    for d in os.listdir(KAGGLE_INPUT):
        candidate = os.path.join(KAGGLE_INPUT, d, slug)
        if os.path.isdir(candidate):
            return candidate
            
    # If not found, return the direct path anyway so the caller's 
    # error message shows the expected path
    return direct_path


# ---------------------------------------------------------------------------
# GOT-10k helpers
# ---------------------------------------------------------------------------

def get_got10k_paths(
    slug: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Resolve train, val, and test directories for the GOT-10k dataset.

    The dataset may be uploaded with different nesting levels.  This
    function probes several common layouts:

    1. ``/kaggle/input/{slug}/train``  (flat)
    2. ``/kaggle/input/{slug}/GOT-10k/train``  (nested under GOT-10k)
    3. Any subdirectory that contains folders named ``train``, ``val``,
       or ``test``.

    GOT-10k splits:
        * train – 9 335 sequences
        * val   – 180 sequences
        * test  – 180 sequences

    Args:
        slug: Dataset slug used in the Kaggle input path.  Defaults to
            ``KAGGLE_DATASET_SLUGS['got10k']``.

    Returns:
        A ``(train_dir, val_dir, test_dir)`` tuple of absolute path
        strings.

    Raises:
        RuntimeError: If the expected directory structure cannot be
            found under the resolved base path.
    """
    if slug is None:
        slug = KAGGLE_DATASET_SLUGS["got10k"]

    base: str = _find_dataset_base(slug)

    if not os.path.isdir(base):
        raise RuntimeError(
            f"GOT-10k base directory not found: {base}\n"
            f"Make sure the dataset is attached to your Kaggle notebook "
            f"with slug '{slug}'.\n"
            f"Contents of /kaggle/input: {os.listdir(KAGGLE_INPUT) if os.path.exists(KAGGLE_INPUT) else 'N/A'}"
        )

    # Strategy 1: flat layout  <base>/val
    if os.path.isdir(os.path.join(base, "val")):
        return (
            os.path.join(base, "train"),
            os.path.join(base, "val"),
            os.path.join(base, "test"),
        )

    # Strategy 2: nested under GOT-10k  <base>/GOT-10k/val
    nested: str = os.path.join(base, "GOT-10k")
    if os.path.isdir(os.path.join(nested, "val")):
        return (
            os.path.join(nested, "train"),
            os.path.join(nested, "val"),
            os.path.join(nested, "test"),
        )

    # Strategy 3: scan one level of subdirectories for val/test folders
    for entry in sorted(os.listdir(base)):
        candidate: str = os.path.join(base, entry)
        if os.path.isdir(candidate):
            if os.path.isdir(os.path.join(candidate, "val")) or os.path.isdir(
                os.path.join(candidate, "test")
            ):
                return (
                    os.path.join(candidate, "train"),
                    os.path.join(candidate, "val"),
                    os.path.join(candidate, "test"),
                )

    raise RuntimeError(
        f"Could not locate GOT-10k train/val/test splits under {base}.\n"
        f"Expected one of:\n"
        f"  {base}/train, {base}/val, {base}/test\n"
        f"  {base}/GOT-10k/train, …/val, …/test\n"
        f"  {base}/<subdir>/train, …/val, …/test\n"
        f"Contents of {base}: {os.listdir(base) if os.path.isdir(base) else '(missing)'}"
    )


# ---------------------------------------------------------------------------
# Kinetics-400 helpers
# ---------------------------------------------------------------------------

def get_kinetics400_csv_path(
    slug: Optional[str] = None,
) -> str:
    """Locate the CSV annotation file for Kinetics-400.

    Searches the dataset directory (and one level of subdirectories) for
    the first ``.csv`` file.

    Args:
        slug: Dataset slug used in the Kaggle input path.  Defaults to
            ``KAGGLE_DATASET_SLUGS['kinetics400']``.

    Returns:
        Absolute path to the discovered CSV file.

    Raises:
        RuntimeError: If no CSV file is found under the base path.
    """
    if slug is None:
        slug = KAGGLE_DATASET_SLUGS["kinetics400"]

    base: str = _find_dataset_base(slug)

    if not os.path.isdir(base):
        raise RuntimeError(
            f"Kinetics-400 base directory not found: {base}\n"
            f"Make sure the dataset is attached to your Kaggle notebook "
            f"with slug '{slug}'."
        )

    # Recursive glob for .csv files
    csv_files = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
    if csv_files:
        return csv_files[0]

    raise RuntimeError(
        f"No CSV file found under {base}.\n"
        f"Contents of {base}: {os.listdir(base)}"
    )


def get_kinetics400_video_dir(
    slug: Optional[str] = None,
) -> Optional[str]:
    """Locate the video directory for Kinetics-400 (if present).

    Looks for a directory structure of the form
    ``{split}/{class_name}/*.mp4`` under the dataset base path.

    Args:
        slug: Dataset slug used in the Kaggle input path.  Defaults to
            ``KAGGLE_DATASET_SLUGS['kinetics400']``.

    Returns:
        The base directory containing video files, or ``None`` if no
        ``.mp4`` files are found.
    """
    if slug is None:
        slug = KAGGLE_DATASET_SLUGS["kinetics400"]

    base: str = _find_dataset_base(slug)

    if not os.path.isdir(base):
        return None

    # Quick check: any .mp4 anywhere under base?
    mp4_files = glob.glob(os.path.join(base, "**", "*.mp4"), recursive=True)
    if mp4_files:
        return base

    return None


def detect_kinetics400_format(
    slug: Optional[str] = None,
) -> str:
    """Detect the storage format of the Kinetics-400 dataset.

    Args:
        slug: Dataset slug used in the Kaggle input path.  Defaults to
            ``KAGGLE_DATASET_SLUGS['kinetics400']``.

    Returns:
        One of:
            * ``'csv'`` – only CSV annotation files are present.
            * ``'video_folder'`` – actual ``.mp4`` video files exist.
            * ``'not_found'`` – the dataset directory does not exist or
              is empty.
    """
    if slug is None:
        slug = KAGGLE_DATASET_SLUGS["kinetics400"]

    base: str = _find_dataset_base(slug)

    if not os.path.isdir(base):
        return "not_found"

    # Check for video files first (more specific).
    if get_kinetics400_video_dir(slug) is not None:
        return "video_folder"

    # Fall back to CSV check.
    try:
        get_kinetics400_csv_path(slug)
        return "csv"
    except RuntimeError:
        pass

    return "not_found"


# ---------------------------------------------------------------------------
# Save / checkpoint helpers
# ---------------------------------------------------------------------------

def get_kaggle_save_dir() -> str:
    """Return the checkpoint save directory for Kaggle runs.

    On Kaggle the directory is created automatically if it does not yet
    exist.  Outside of Kaggle the path is still returned (but not
    created) so that callers can handle local vs. remote logic
    themselves.

    Returns:
        ``/kaggle/working/checkpoints``
    """
    save_dir: str = os.path.join(KAGGLE_WORKING, "checkpoints")

    if is_kaggle():
        os.makedirs(save_dir, exist_ok=True)

    return save_dir
