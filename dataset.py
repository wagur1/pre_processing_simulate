"""
dataset.py — Video Frame Dataset for Preprocessing Framework
=============================================================
Supports two data sources:

  1. **HuggingFace Kinetics-400** — loaded via `datasets.load_dataset()`
     Each sample has a 'video' field (decoded frames) and a 'label' field.

  2. **GOT-10k** (local folders) — sequences of JPEG frames on disk.
     Structure: got10k_root/val/GOT-10k_Val_000001/00000001.jpg …

Both are wrapped into a unified PyTorch Dataset that outputs
(clip, label) tensors with an 80/20 stratified train/test split and
data augmentation for the training split.

Paper: "A Preprocessing Framework for Video Machine Vision under Compression"
"""

import os
import csv
import glob
import random
import math
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

try:
    import cv2
    _USE_CV2 = True
except ImportError:
    _USE_CV2 = False


# ======================================================================== #
#              1. HuggingFace Kinetics-400 Dataset                         #
# ======================================================================== #

class HFKinetics400Dataset(Dataset):
    """PyTorch Dataset wrapping a HuggingFace `datasets` split of Kinetics-400.

    Usage
    -----
    >>> from datasets import load_dataset
    >>> hf_val = load_dataset("video-dataset/kinetics400", split="validation")
    >>> ds = HFKinetics400Dataset(hf_val, num_frames=8)

    The HF dataset is expected to have:
      - A 'video' column  (decoded video frames or path)
      - A 'label' column  (integer class index)
    """

    def __init__(
        self,
        hf_dataset,
        num_frames: int = 8,
        frame_stride: int = 2,
        frame_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.frame_size = frame_size
        self.augment = augment

        # Detect column names (different HF uploads may use different names)
        cols = set(hf_dataset.column_names)
        self.video_col = "video" if "video" in cols else "video_path"
        self.label_col = "label" if "label" in cols else "labels"

        # Build a label list for stratified splitting
        self.labels: List[int] = []
        for i in range(len(hf_dataset)):
            self.labels.append(hf_dataset[i][self.label_col])

        # Unique sorted class list (for compatibility with train.py)
        unique_labels = sorted(set(self.labels))
        self.classes = [str(l) for l in unique_labels]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _extract_frames(self, video_data) -> Optional[torch.Tensor]:
        """Extract frames from the HF video field.

        The 'video' field from HuggingFace datasets is typically a list of
        PIL images (decoded frames).  We sample `num_frames` with stride.
        """
        # If it's a list of PIL images (most common for HF video datasets)
        if isinstance(video_data, (list, tuple)):
            pil_frames = video_data
        # If the HF dataset wraps it in a dict with a 'path' or frames key
        elif isinstance(video_data, dict):
            if "path" in video_data:
                return self._decode_from_path(video_data["path"])
            # Some datasets store decoded frames directly
            pil_frames = video_data.get("frames", video_data.get("video", []))
        else:
            # It might be a file path string
            if isinstance(video_data, str) and os.path.isfile(video_data):
                return self._decode_from_path(video_data)
            return None

        total = len(pil_frames)
        needed = self.num_frames * self.frame_stride
        if total < needed:
            # If not enough frames, reduce stride or repeat
            if total >= self.num_frames:
                actual_stride = max(1, total // self.num_frames)
                indices = list(range(0, self.num_frames * actual_stride, actual_stride))
            else:
                # Repeat last frame to fill
                indices = list(range(total)) + [total - 1] * (self.num_frames - total)
        else:
            start = random.randint(0, total - needed)
            indices = list(range(start, start + needed, self.frame_stride))

        frames = []
        for idx in indices[:self.num_frames]:
            pil_img = pil_frames[idx]
            # Convert PIL → tensor (C, H, W) float [0, 1]
            t = TF.to_tensor(pil_img)  # auto-converts PIL to (C, H, W) float
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)  # grayscale → RGB
            elif t.shape[0] == 4:
                t = t[:3]  # RGBA → RGB
            frames.append(t)

        return torch.stack(frames)  # (T, C, H, W)

    def _decode_from_path(self, video_path: str) -> Optional[torch.Tensor]:
        """Decode frames from a video file path using OpenCV."""
        if not _USE_CV2:
            return None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        needed = self.num_frames * self.frame_stride
        if total < needed:
            cap.release()
            return None
        start = random.randint(0, total - needed)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for i in range(needed):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            if i % self.frame_stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(t)
        cap.release()
        return torch.stack(frames) if len(frames) == self.num_frames else None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.hf_dataset[idx]
        label = sample[self.label_col]
        video_data = sample[self.video_col]

        frames = self._extract_frames(video_data)
        if frames is None:
            frames = torch.zeros(self.num_frames, 3, self.frame_size[0], self.frame_size[1])

        # Apply spatial transforms (temporally consistent)
        frames = self._apply_transforms(frames)
        return frames, label

    def _apply_transforms(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply spatial transforms consistently across all frames in the clip."""
        if self.augment:
            # Same random crop params for all frames in the clip
            params = T.RandomResizedCrop.get_params(
                frames[0], scale=(0.8, 1.0), ratio=(0.75, 1.33)
            )
            do_flip = random.random() < 0.5
            transformed = []
            for f in frames:
                f = TF.resized_crop(f, *params, self.frame_size)
                if do_flip:
                    f = TF.hflip(f)
                transformed.append(f)
        else:
            transformed = []
            for f in frames:
                f = TF.resize(f, int(self.frame_size[0] * 1.15))
                f = TF.center_crop(f, self.frame_size)
                transformed.append(f)
        return torch.stack(transformed)  # (T, C, H, W)


# ======================================================================== #
#              2. GOT-10k Image-Sequence Dataset                           #
# ======================================================================== #

class GOT10kDataset(Dataset):
    """PyTorch Dataset for GOT-10k (val or test split).

    GOT-10k stores each video as a directory of JPEG frames:
        got10k_root/
          val/
            GOT-10k_Val_000001/
              00000001.jpg
              00000002.jpg
              ...
            GOT-10k_Val_000180/
            list.txt
          test/
            GOT-10k_Test_000001/
              ...
            list.txt

    Since GOT-10k is an object-tracking dataset (not classification),
    sequences are grouped into ``num_pseudo_classes`` pseudo-classes
    via round-robin assignment.  This ensures each pseudo-class has
    multiple samples so that an 80/20 stratified split produces
    non-empty train AND test subsets.

    Parameters
    ----------
    root_dir : str
        Path to the split folder, e.g., ``/data/got10k/val``.
    num_pseudo_classes : int
        Number of pseudo-classes to create (default 20).
    num_frames, frame_stride, frame_size, augment : same as above.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 8,
        frame_stride: int = 2,
        frame_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        num_pseudo_classes: int = 20,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.frame_size = frame_size
        self.augment = augment

        # ---- Discover sequences ----
        # Use list.txt if available, otherwise scan subdirectories
        list_file = os.path.join(root_dir, "list.txt")
        if os.path.isfile(list_file):
            with open(list_file, "r") as f:
                seq_names = [line.strip() for line in f if line.strip()]
        else:
            seq_names = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )

        # ---- Build samples ----
        # Group sequences into pseudo-classes via round-robin so that
        # each class has multiple samples (needed for stratified split).
        valid_seqs: List[str] = []
        for seq_name in seq_names:
            seq_dir = os.path.join(root_dir, seq_name)
            if os.path.isdir(seq_dir):
                frame_files = self._list_frames(seq_dir)
                if len(frame_files) >= self.num_frames:
                    valid_seqs.append(seq_name)

        if len(valid_seqs) == 0:
            raise RuntimeError(
                f"No valid GOT-10k sequences found in '{root_dir}'. "
                f"Expected subdirectories with JPEG frames inside."
            )

        # Assign pseudo-labels: sequence i → class (i % num_pseudo_classes)
        actual_num_classes = min(num_pseudo_classes, len(valid_seqs))
        self.sequences: List[Tuple[str, int]] = []
        for i, seq_name in enumerate(valid_seqs):
            seq_dir = os.path.join(root_dir, seq_name)
            pseudo_label = i % actual_num_classes
            self.sequences.append((seq_dir, pseudo_label))

        self.classes = [f"pseudo_class_{i}" for i in range(actual_num_classes)]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # For compatibility with build_splits (needs .samples and .labels)
        self.samples = self.sequences
        self.labels = [label for _, label in self.sequences]

    def _list_frames(self, seq_dir: str) -> List[str]:
        """Return sorted list of image file paths in a sequence directory."""
        files = []
        for f in sorted(os.listdir(seq_dir)):
            if os.path.splitext(f)[1].lower() in self.IMG_EXTS:
                files.append(os.path.join(seq_dir, f))
        return files

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq_dir, label = self.sequences[idx]
        frame_files = self._list_frames(seq_dir)

        total = len(frame_files)
        needed = self.num_frames * self.frame_stride

        # Sample frame indices
        if total >= needed:
            start = random.randint(0, total - needed)
            indices = list(range(start, start + needed, self.frame_stride))
        elif total >= self.num_frames:
            stride = max(1, total // self.num_frames)
            indices = list(range(0, self.num_frames * stride, stride))
        else:
            indices = list(range(total)) + [total - 1] * (self.num_frames - total)

        indices = indices[:self.num_frames]

        # Load frames — resize each to a uniform size immediately so
        # torch.stack works even when a sequence has mixed resolutions.
        intermediate_h = int(self.frame_size[0] * 1.15)
        intermediate_w = int(self.frame_size[1] * 1.15)

        frames = []
        for i in indices:
            fpath = frame_files[i]
            if _USE_CV2:
                img = cv2.imread(fpath)
                if img is None:
                    frames.append(torch.zeros(3, intermediate_h, intermediate_w))
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            else:
                from PIL import Image
                img = Image.open(fpath).convert("RGB")
                t = TF.to_tensor(img)
            # Resize to uniform intermediate size before stacking
            t = TF.resize(t, [intermediate_h, intermediate_w])
            frames.append(t)

        frames = torch.stack(frames)  # (T, C, H, W)

        # Apply transforms
        frames = self._apply_transforms(frames)
        return frames, label

    def _apply_transforms(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply spatial transforms consistently across all frames."""
        if self.augment:
            params = T.RandomResizedCrop.get_params(
                frames[0], scale=(0.8, 1.0), ratio=(0.75, 1.33)
            )
            do_flip = random.random() < 0.5
            transformed = []
            for f in frames:
                f = TF.resized_crop(f, *params, self.frame_size)
                if do_flip:
                    f = TF.hflip(f)
                transformed.append(f)
        else:
            transformed = []
            for f in frames:
                f = TF.resize(f, int(self.frame_size[0] * 1.15))
                f = TF.center_crop(f, self.frame_size)
                transformed.append(f)
        return torch.stack(transformed)


# ======================================================================== #
#               80/20 Dynamic Train / Test Split Utility                   #
# ======================================================================== #

def build_splits_from_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    seed: int = 42,
    augment_train: bool = True,
) -> Tuple[Subset, Subset]:
    """Create an 80/20 stratified train/test split from any dataset.

    The split is *stratified*: each class contributes proportionally to
    both subsets, preventing hidden label distribution issues.

    Parameters
    ----------
    dataset : Dataset
        Must have a `labels` attribute (List[int]) for stratification.
    train_ratio : float
        Fraction of data for training (default 0.8).
    seed : int
        Random seed for reproducibility.
    augment_train : bool
        If True, creates a *separate* augmented copy for training indices.
        Only works if the dataset has an `augment` attribute.
    """
    labels = dataset.labels  # type: ignore

    class_indices: dict = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []

    for cls_label in sorted(class_indices.keys()):
        idxs = class_indices[cls_label][:]
        rng.shuffle(idxs)
        split_point = max(1, int(len(idxs) * train_ratio))
        train_indices.extend(idxs[:split_point])
        test_indices.extend(idxs[split_point:])

    # If we can create an augmented copy for training, do so
    if augment_train and hasattr(dataset, 'augment'):
        import copy
        train_ds = copy.copy(dataset)
        train_ds.augment = True
        test_ds = dataset
        test_ds.augment = False
    else:
        train_ds = dataset
        test_ds = dataset

    return Subset(train_ds, train_indices), Subset(test_ds, test_indices)


# ======================================================================== #
#                     High-Level Builder Functions                         #
# ======================================================================== #

def build_kinetics400_splits(
    split: str = "validation",
    num_frames: int = 8,
    frame_stride: int = 2,
    frame_size: Tuple[int, int] = (224, 224),
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[Subset, Subset, int]:
    """Load Kinetics-400 from HuggingFace and create 80/20 train/test splits.

    Parameters
    ----------
    split : str
        HuggingFace split name ("validation" or "test").

    Returns
    -------
    train_subset, test_subset, num_classes
    """
    from datasets import load_dataset

    print(f"[INFO] Loading Kinetics-400 '{split}' from HuggingFace …")
    hf_data = load_dataset("video-dataset/kinetics400", split=split)
    print(f"[INFO] Loaded {len(hf_data)} samples")

    # Non-augmented base for splitting
    base_ds = HFKinetics400Dataset(
        hf_data,
        num_frames=num_frames,
        frame_stride=frame_stride,
        frame_size=frame_size,
        augment=False,
    )

    # Augmented copy for training
    train_ds = HFKinetics400Dataset(
        hf_data,
        num_frames=num_frames,
        frame_stride=frame_stride,
        frame_size=frame_size,
        augment=True,
    )

    # Stratified split
    class_indices: dict = defaultdict(list)
    for idx, label in enumerate(base_ds.labels):
        class_indices[label].append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []

    for cls_label in sorted(class_indices.keys()):
        idxs = class_indices[cls_label][:]
        rng.shuffle(idxs)
        split_point = max(1, int(len(idxs) * train_ratio))
        train_indices.extend(idxs[:split_point])
        test_indices.extend(idxs[split_point:])

    num_classes = len(base_ds.classes)

    return (
        Subset(train_ds, train_indices),
        Subset(base_ds, test_indices),
        num_classes,
    )


def build_got10k_train_test(
    train_dir: str,
    test_dir: str,
    num_frames: int = 8,
    frame_stride: int = 2,
    frame_size: Tuple[int, int] = (224, 224),
    num_pseudo_classes: int = 20,
) -> Tuple[Dataset, Dataset, int]:
    """Build GOT-10k train and test datasets from separate folders.

    Uses the GOT-10k **val** folder for training (with augmentation)
    and the **test** folder for evaluation (no augmentation).

    Parameters
    ----------
    train_dir : str
        Path to training sequences (e.g., ``/data/got10k/val``).
    test_dir : str
        Path to test sequences (e.g., ``/data/got10k/test``).

    Returns
    -------
    train_ds, test_ds, num_classes
    """
    train_ds = GOT10kDataset(
        train_dir, num_frames, frame_stride, frame_size,
        augment=True, num_pseudo_classes=num_pseudo_classes,
    )
    test_ds = GOT10kDataset(
        test_dir, num_frames, frame_stride, frame_size,
        augment=False, num_pseudo_classes=num_pseudo_classes,
    )

    num_classes = max(len(train_ds.classes), len(test_ds.classes))
    print(f"[INFO] GOT-10k: train={len(train_ds)} sequences ({train_dir}), "
          f"test={len(test_ds)} sequences ({test_dir}), "
          f"{num_classes} pseudo-classes")

    return train_ds, test_ds, num_classes


# ======================================================================== #
#        3. Kaggle Video-Folder Kinetics-400 Dataset                       #
# ======================================================================== #

class KaggleKinetics400Dataset(Dataset):
    """PyTorch Dataset for Kinetics-400 stored as video files in class folders.

    Supports any Kaggle dataset organized as:
        root_dir/
          {class_name_1}/
            video_001.mp4
            video_002.avi
            ...
          {class_name_2}/
            ...

    Each class subfolder name becomes the label.  Video files are decoded
    frame-by-frame using OpenCV.

    Parameters
    ----------
    root_dir : str
        Path to the split folder containing class subdirectories.
    num_frames : int
        Number of frames to sample per clip.
    frame_stride : int
        Stride between sampled frames.
    frame_size : Tuple[int, int]
        Target spatial resolution (H, W).
    augment : bool
        If True, apply random augmentation.
    max_samples_per_class : int, optional
        If set, limit the number of videos per class (useful for large datasets).
    """

    VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 8,
        frame_stride: int = 2,
        frame_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        max_samples_per_class: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.frame_size = frame_size
        self.augment = augment

        if not _USE_CV2:
            raise RuntimeError(
                "OpenCV (cv2) is required for KaggleKinetics400Dataset. "
                "Install it with: pip install opencv-python"
            )

        # ---- Discover classes and videos ----
        class_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if len(class_dirs) == 0:
            raise RuntimeError(
                f"No class subdirectories found in '{root_dir}'. "
                f"Expected folder-per-class structure with video files."
            )

        self.classes = class_dirs
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Build list of (video_path, label) tuples
        self.samples: List[Tuple[str, int]] = []
        for cls_name in class_dirs:
            cls_dir = os.path.join(root_dir, cls_name)
            label = self.class_to_idx[cls_name]
            videos = sorted([
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in self.VIDEO_EXTS
            ])
            if max_samples_per_class:
                videos = videos[:max_samples_per_class]
            for vpath in videos:
                self.samples.append((vpath, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No video files found in '{root_dir}'. "
                f"Supported formats: {self.VIDEO_EXTS}"
            )

        self.labels = [label for _, label in self.samples]
        print(f"[INFO] KaggleKinetics400: {len(self.samples)} videos, "
              f"{len(self.classes)} classes from '{root_dir}'")

    def __len__(self) -> int:
        return len(self.samples)

    def _decode_video(self, video_path: str) -> Optional[torch.Tensor]:
        """Decode frames from a video file using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        needed = self.num_frames * self.frame_stride

        if total < self.num_frames:
            # Too few frames — read all and pad
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                all_frames.append(t)
            cap.release()
            if len(all_frames) == 0:
                return None
            # Pad by repeating last frame
            while len(all_frames) < self.num_frames:
                all_frames.append(all_frames[-1])
            return torch.stack(all_frames[:self.num_frames])

        # Enough frames — sample with stride
        if total >= needed:
            start = random.randint(0, total - needed)
            indices = list(range(start, start + needed, self.frame_stride))
        else:
            stride = max(1, total // self.num_frames)
            indices = list(range(0, self.num_frames * stride, stride))

        indices = indices[:self.num_frames]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(t)
        cap.release()

        if len(frames) != self.num_frames:
            return None
        return torch.stack(frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = self._decode_video(video_path)
        if frames is None:
            frames = torch.zeros(self.num_frames, 3, self.frame_size[0], self.frame_size[1])
        frames = self._apply_transforms(frames)
        return frames, label

    def _apply_transforms(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply spatial transforms consistently across all frames."""
        # Resize all frames to uniform size first
        intermediate_h = int(self.frame_size[0] * 1.15)
        intermediate_w = int(self.frame_size[1] * 1.15)
        resized = []
        for f in frames:
            f = TF.resize(f, [intermediate_h, intermediate_w])
            resized.append(f)
        frames = torch.stack(resized)

        if self.augment:
            params = T.RandomResizedCrop.get_params(
                frames[0], scale=(0.8, 1.0), ratio=(0.75, 1.33)
            )
            do_flip = random.random() < 0.5
            transformed = []
            for f in frames:
                f = TF.resized_crop(f, *params, self.frame_size)
                if do_flip:
                    f = TF.hflip(f)
                transformed.append(f)
        else:
            transformed = []
            for f in frames:
                f = TF.center_crop(f, self.frame_size)
                transformed.append(f)
        return torch.stack(transformed)


# ======================================================================== #
#        4. CSV-Based Kinetics-400 Dataset (Metadata Only)                 #
# ======================================================================== #

class CSVKinetics400Dataset(Dataset):
    """PyTorch Dataset for Kinetics-400 from a CSV metadata file.

    This handles Kaggle datasets like ``nikiforosvagenas/kinetics-400``
    which contain only a CSV file with columns such as:
        label, youtube_id, time_start, time_end, split, is_cc

    Since actual video files are not included, this dataset generates
    **synthetic placeholder clips** with per-class characteristics for
    architecture validation and training pipeline testing.  For real
    training, use ``KaggleKinetics400Dataset`` or ``HFKinetics400Dataset``.

    Alternatively, if ``video_dir`` is provided, the dataset will attempt
    to find downloaded video files matching the youtube_id.

    Parameters
    ----------
    csv_path : str
        Path to the CSV metadata file.
    split : str
        Which split to use (filters the 'split' column if present).
        Common values: 'train', 'validate', 'test'.
    num_frames : int
        Number of frames per clip.
    frame_size : Tuple[int, int]
        Target spatial resolution (H, W).
    augment : bool
        If True, apply augmentation.
    video_dir : str, optional
        Directory containing downloaded video files named as
        ``{youtube_id}_{time_start:06d}_{time_end:06d}.mp4``.
    max_samples : int, optional
        Maximum number of samples to load (useful for testing).
    """

    def __init__(
        self,
        csv_path: str,
        split: Optional[str] = None,
        num_frames: int = 8,
        frame_stride: int = 2,
        frame_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        video_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.frame_size = frame_size
        self.augment = augment
        self.video_dir = video_dir

        # ---- Parse CSV ----
        self.samples: List[Dict] = []
        self._parse_csv(csv_path, split, max_samples)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in '{csv_path}' "
                f"(split={split}). Check the CSV format and column names."
            )

        # ---- Build class mapping ----
        unique_labels = sorted(set(s["label"] for s in self.samples))
        self.classes = unique_labels
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Convert string labels to integer indices
        self.labels: List[int] = []
        for s in self.samples:
            s["label_idx"] = self.class_to_idx[s["label"]]
            self.labels.append(s["label_idx"])

        print(f"[INFO] CSVKinetics400: {len(self.samples)} samples, "
              f"{len(self.classes)} classes from '{csv_path}'"
              f" (split={split})")

    def _parse_csv(
        self, csv_path: str, split: Optional[str], max_samples: Optional[int]
    ) -> None:
        """Parse the CSV file and populate self.samples."""
        with open(csv_path, "r", encoding="utf-8") as f:
            # Auto-detect dialect
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(f, dialect=dialect)
            except csv.Error:
                reader = csv.DictReader(f)

            # Normalize column names (handle different CSV formats)
            for row in reader:
                # Normalize keys to lowercase, strip whitespace
                normalized = {k.strip().lower().replace(" ", "_"): v.strip()
                              for k, v in row.items() if k is not None}

                # Extract label (try multiple common column names)
                label = (normalized.get("label") or
                         normalized.get("class") or
                         normalized.get("action") or
                         normalized.get("category") or "")

                if not label:
                    continue

                # Filter by split if requested
                row_split = (normalized.get("split") or
                             normalized.get("subset") or "")
                if split and row_split and row_split.lower() != split.lower():
                    continue

                # Extract video info
                youtube_id = (normalized.get("youtube_id") or
                              normalized.get("id") or
                              normalized.get("video_id") or "")
                time_start = normalized.get("time_start", "0")
                time_end = normalized.get("time_end", "10")

                self.samples.append({
                    "label": label,
                    "youtube_id": youtube_id,
                    "time_start": time_start,
                    "time_end": time_end,
                })

                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def _try_load_video(self, sample: Dict) -> Optional[torch.Tensor]:
        """Attempt to load video from video_dir if available."""
        if not self.video_dir or not _USE_CV2:
            return None

        yt_id = sample["youtube_id"]
        if not yt_id:
            return None

        # Try common filename patterns
        patterns = [
            f"{yt_id}*.mp4", f"{yt_id}*.avi", f"{yt_id}*.mkv",
        ]
        for pattern in patterns:
            matches = glob.glob(os.path.join(self.video_dir, pattern))
            if matches:
                cap = cv2.VideoCapture(matches[0])
                if cap.isOpened():
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    needed = self.num_frames * self.frame_stride
                    if total >= self.num_frames:
                        if total >= needed:
                            start = random.randint(0, total - needed)
                            indices = list(range(start, start + needed, self.frame_stride))
                        else:
                            stride = max(1, total // self.num_frames)
                            indices = list(range(0, self.num_frames * stride, stride))
                        indices = indices[:self.num_frames]
                        frames = []
                        for idx in indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                                frames.append(t)
                        cap.release()
                        if len(frames) == self.num_frames:
                            return torch.stack(frames)
                    cap.release()
        return None

    def _generate_synthetic_clip(self, label_idx: int) -> torch.Tensor:
        """Generate a deterministic synthetic clip based on the class label.

        Uses the label index to create visually distinguishable patterns
        for each class — useful for validating the training pipeline
        and architecture without real video data.
        """
        H, W = self.frame_size
        # Use label to seed deterministic color/pattern
        rng = random.Random(label_idx * 1000 + 42)

        # Base color varies by class
        base_r = rng.uniform(0.1, 0.9)
        base_g = rng.uniform(0.1, 0.9)
        base_b = rng.uniform(0.1, 0.9)

        frames = []
        for t_idx in range(self.num_frames):
            # Add temporal variation (simulates motion)
            time_offset = t_idx / self.num_frames
            frame = torch.zeros(3, H, W)
            frame[0] = base_r + 0.1 * math.sin(2 * math.pi * time_offset)
            frame[1] = base_g + 0.1 * math.cos(2 * math.pi * time_offset)
            frame[2] = base_b + 0.05 * math.sin(4 * math.pi * time_offset)

            # Add spatial gradient (different per class)
            for i in range(H):
                for c in range(3):
                    frame[c, i, :] += (i / H) * 0.2 * (label_idx % 5) / 5.0

            frame = frame.clamp(0, 1)
            frames.append(frame)

        return torch.stack(frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        label_idx = sample["label_idx"]

        # Try loading real video first
        frames = self._try_load_video(sample)

        if frames is None:
            # Fall back to synthetic data
            frames = self._generate_synthetic_clip(label_idx)

        frames = self._apply_transforms(frames)
        return frames, label_idx

    def _apply_transforms(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply spatial transforms consistently across all frames."""
        if self.augment:
            params = T.RandomResizedCrop.get_params(
                frames[0], scale=(0.8, 1.0), ratio=(0.75, 1.33)
            )
            do_flip = random.random() < 0.5
            transformed = []
            for f in frames:
                f = TF.resized_crop(f, *params, self.frame_size)
                if do_flip:
                    f = TF.hflip(f)
                transformed.append(f)
        else:
            transformed = []
            for f in frames:
                f = TF.resize(f, int(self.frame_size[0] * 1.15))
                f = TF.center_crop(f, self.frame_size)
                transformed.append(f)
        return torch.stack(transformed)


# ======================================================================== #
#            Kaggle High-Level Builder Functions                            #
# ======================================================================== #

def _find_dataset_base(slug: str) -> str:
    """Find the base directory for a dataset slug, handling nested Kaggle mounts up to 3 levels deep."""
    kaggle_input = "/kaggle/input"
    direct_path = os.path.join(kaggle_input, slug)
    if os.path.isdir(direct_path):
        return direct_path
        
    if os.path.isdir(kaggle_input):
        for d1 in os.listdir(kaggle_input):
            p1 = os.path.join(kaggle_input, d1)
            if not os.path.isdir(p1): continue
            if d1 == slug: return p1
            
            for d2 in os.listdir(p1):
                p2 = os.path.join(p1, d2)
                if not os.path.isdir(p2): continue
                if d2 == slug: return p2
                
                for d3 in os.listdir(p2):
                    p3 = os.path.join(p2, d3)
                    if not os.path.isdir(p3): continue
                    if d3 == slug: return p3

    return direct_path


def _get_kaggle_input_tree() -> str:
    """Return a string representation of the /kaggle/input directory tree up to 2 levels deep."""
    tree = []
    if os.path.exists("/kaggle/input"):
        for d1 in os.listdir("/kaggle/input"):
            tree.append(f"/{d1}")
            p1 = os.path.join("/kaggle/input", d1)
            if os.path.isdir(p1):
                try:
                    for d2 in os.listdir(p1):
                        tree.append(f"  /{d1}/{d2}")
                except OSError:
                    pass
    return "\n".join(tree) if tree else "Empty or not found"


def _find_got10k_subdir(base_path: str, target: str) -> Optional[str]:
    """Auto-detect GOT-10k subdirectory, handling nested folder structures.

    Kaggle datasets may have extra nesting like:
        /kaggle/input/got10k/GOT-10k/val/  instead of
        /kaggle/input/got10k/val/
    """
    # Direct path
    direct = os.path.join(base_path, target)
    if os.path.isdir(direct):
        return direct

    # One level of nesting
    for sub in os.listdir(base_path):
        nested = os.path.join(base_path, sub, target)
        if os.path.isdir(nested):
            return nested

    # Two levels of nesting
    for sub1 in os.listdir(base_path):
        sub1_path = os.path.join(base_path, sub1)
        if os.path.isdir(sub1_path):
            for sub2 in os.listdir(sub1_path):
                nested = os.path.join(sub1_path, sub2, target)
                if os.path.isdir(nested):
                    return nested

    return None


def build_kaggle_got10k_train_test(
    kaggle_slug: str = "got10k",
    num_frames: int = 8,
    frame_stride: int = 2,
    frame_size: Tuple[int, int] = (224, 224),
    num_pseudo_classes: int = 20,
    train_split: str = "val",
    test_split: str = "test",
) -> Tuple[Dataset, Dataset, int]:
    """Build GOT-10k train and test datasets from a Kaggle dataset.

    Automatically resolves paths from /kaggle/input/{slug}/ and handles
    nested directory structures commonly found in Kaggle uploads.

    Parameters
    ----------
    kaggle_slug : str
        Kaggle dataset slug (default: 'got10k' for abhimanyukarshni/got10k).
    train_split : str
        Which split folder to use for training (default: 'val').
    test_split : str
        Which split folder to use for testing (default: 'test').

    Returns
    -------
    train_ds, test_ds, num_classes
    """
    base_path = _find_dataset_base(kaggle_slug)

    if not os.path.isdir(base_path):
        tree = _get_kaggle_input_tree()
        raise RuntimeError(
            f"Kaggle dataset not found at '{base_path}'.\n"
            f"Make sure you've added the dataset '{kaggle_slug}' to your Kaggle notebook.\n"
            f"Directory structure of /kaggle/input:\n{tree}"
        )

    train_dir = _find_got10k_subdir(base_path, train_split)
    test_dir = _find_got10k_subdir(base_path, test_split)

    if train_dir is None:
        raise RuntimeError(
            f"Could not find '{train_split}/' directory in '{base_path}'. "
            f"Searched direct path and nested subdirectories. "
            f"Available contents: {os.listdir(base_path)}"
        )
    if test_dir is None:
        raise RuntimeError(
            f"Could not find '{test_split}/' directory in '{base_path}'. "
            f"Searched direct path and nested subdirectories. "
            f"Available contents: {os.listdir(base_path)}"
        )

    print(f"[INFO] Kaggle GOT-10k paths resolved:")
    print(f"[INFO]   Train: {train_dir}")
    print(f"[INFO]   Test : {test_dir}")

    return build_got10k_train_test(
        train_dir=train_dir,
        test_dir=test_dir,
        num_frames=num_frames,
        frame_stride=frame_stride,
        frame_size=frame_size,
        num_pseudo_classes=num_pseudo_classes,
    )


def _find_csv_file(base_path: str) -> Optional[str]:
    """Find a CSV file in the given directory (recursive)."""
    for root, dirs, files in os.walk(base_path):
        for f in sorted(files):
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)
    return None


def _find_video_folder(base_path: str) -> Optional[str]:
    """Find a directory containing class-organized video files."""
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

    # Check if base_path itself has class subdirs with videos
    for entry in os.listdir(base_path):
        entry_path = os.path.join(base_path, entry)
        if os.path.isdir(entry_path):
            # Check if this subdir contains video files
            for f in os.listdir(entry_path):
                if os.path.splitext(f)[1].lower() in video_exts:
                    return base_path
            # Check one level deeper (e.g., base/train/{class}/*.mp4)
            for sub_entry in os.listdir(entry_path):
                sub_path = os.path.join(entry_path, sub_entry)
                if os.path.isdir(sub_path):
                    for f in os.listdir(sub_path):
                        if os.path.splitext(f)[1].lower() in video_exts:
                            return entry_path
    return None


def build_kaggle_kinetics400_splits(
    kaggle_slug: str = "kinetics-400",
    split: Optional[str] = None,
    num_frames: int = 8,
    frame_stride: int = 2,
    frame_size: Tuple[int, int] = (224, 224),
    train_ratio: float = 0.8,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[Subset, Subset, int]:
    """Build Kinetics-400 train/test splits from a Kaggle dataset.

    Automatically detects whether the dataset contains:
      - Video files in class folders → uses ``KaggleKinetics400Dataset``
      - A CSV metadata file → uses ``CSVKinetics400Dataset``

    Parameters
    ----------
    kaggle_slug : str
        Kaggle dataset slug (default: 'kinetics-400').
    split : str, optional
        Filter CSV by split column (e.g., 'train', 'validate', 'test').
    max_samples : int, optional
        Limit total samples (useful for testing).

    Returns
    -------
    train_subset, test_subset, num_classes
    """
    base_path = _find_dataset_base(kaggle_slug)

    if not os.path.isdir(base_path):
        tree = _get_kaggle_input_tree()
        raise RuntimeError(
            f"Kaggle dataset not found at '{base_path}'.\n"
            f"Make sure you've added the dataset '{kaggle_slug}' to your Kaggle notebook.\n"
            f"Directory structure of /kaggle/input:\n{tree}"
        )

    # ---- Auto-detect format ----
    video_dir = _find_video_folder(base_path)
    csv_path = _find_csv_file(base_path)

    if video_dir is not None:
        # Format: class folders with video files
        print(f"[INFO] Detected video folder format at: {video_dir}")
        base_ds = KaggleKinetics400Dataset(
            video_dir,
            num_frames=num_frames,
            frame_stride=frame_stride,
            frame_size=frame_size,
            augment=False,
            max_samples_per_class=max_samples,
        )
    elif csv_path is not None:
        # Format: CSV metadata file
        print(f"[INFO] Detected CSV metadata format at: {csv_path}")
        print(f"[WARN] No video files found — using synthetic clips for "
              f"pipeline validation. For real training, use a dataset "
              f"with actual video files.")
        base_ds = CSVKinetics400Dataset(
            csv_path,
            split=split,
            num_frames=num_frames,
            frame_stride=frame_stride,
            frame_size=frame_size,
            augment=False,
            max_samples=max_samples,
        )
    else:
        raise RuntimeError(
            f"Could not detect Kinetics-400 data format in '{base_path}'. "
            f"Expected either class-folder structure with video files "
            f"or a CSV metadata file. "
            f"Contents: {os.listdir(base_path)}"
        )

    # ---- Create augmented copy for training ----
    import copy
    train_ds = copy.copy(base_ds)
    train_ds.augment = True
    test_ds = base_ds
    test_ds.augment = False

    # ---- Stratified split ----
    train_sub, test_sub = build_splits_from_dataset(
        base_ds, train_ratio=train_ratio, seed=seed, augment_train=True,
    )

    num_classes = len(base_ds.classes)
    return train_sub, test_sub, num_classes

