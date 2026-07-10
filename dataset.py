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
import glob
import random
import math
from collections import defaultdict
from typing import Tuple, List, Optional

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
