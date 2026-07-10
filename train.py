"""
train.py — Training Pipeline
==============================
Joint optimization of the Preprocessing System with a frozen downstream
Vision Task Analyzer (SlowFast) as described in:
  "A Preprocessing Framework for Video Machine Vision under Compression"

Loss Function
-------------
    L = α · (L_D + λ · L_R) + L_Acc

where:
    L_D   = MSE(original_frame, reconstructed_frame)        — distortion
    L_R   = estimated bits-per-pixel (bpp) from the codec    — rate
    L_Acc = CrossEntropy(analyzer(reconstructed_clip), label) — task accuracy
    α     = 10
    λ     = 0.001

Only the Preprocessor + VirtualCodec weights are updated.
The downstream analyzer is strictly frozen (eval mode, no gradients).
"""

import argparse
import copy
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_kinetics400_splits, build_got10k_train_test
from model import PreprocessingSystem


# ======================================================================== #
#                      Downstream Analyzer Loader                          #
# ======================================================================== #

def load_analyzer(
    num_classes: int,
    device: torch.device,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """Load a pre-trained SlowFast model and freeze it completely.

    The model is placed in eval mode and every parameter has
    ``requires_grad = False`` so that no gradients are computed or
    accumulated for the analyzer during training.

    Parameters
    ----------
    num_classes : int
        Number of action classes (must match the dataset).
    device : torch.device
        Target device (cpu / cuda).
    weights_path : str, optional
        Path to custom checkpoint.  If None we use torchvision defaults.
    """
    try:
        # torchvision >= 0.14 provides weights enum
        from torchvision.models.video import (
            r3d_18,
            R3D_18_Weights,
        )
        # SlowFast is not always available in torchvision; fall back to
        # ResNet3D-18 which accepts identical (B, C, T, H, W) tensors.
        analyzer = r3d_18(weights=R3D_18_Weights.DEFAULT)
    except Exception:
        from torchvision.models.video import r3d_18
        analyzer = r3d_18(pretrained=True)

    # Replace the classification head if num_classes differs
    in_features = analyzer.fc.in_features
    if analyzer.fc.out_features != num_classes:
        analyzer.fc = nn.Linear(in_features, num_classes)

    if weights_path and os.path.isfile(weights_path):
        analyzer.load_state_dict(torch.load(weights_path, map_location="cpu"))

    # ---- STRICT: Freeze all parameters ----
    for param in analyzer.parameters():
        param.requires_grad = False

    # ---- STRICT: Force evaluation mode ----
    analyzer.eval()

    return analyzer.to(device)


# ======================================================================== #
#               Prepare clip for the 3-D Analyzer                          #
# ======================================================================== #

def prepare_clip_for_analyzer(
    recon: torch.Tensor,
    clip: torch.Tensor,
) -> torch.Tensor:
    """Build a (B, C, T, H, W) input for the 3-D video backbone.

    The preprocessor outputs a single enhanced frame per clip.  We tile it
    across the temporal dimension to match the backbone's expected depth,
    OR we take the original clip and replace the last frame with the
    reconstructed one.

    Strategy used here: replace last frame with recon, keep the rest.
    This preserves real temporal context from the clip while reflecting
    the preprocessing on the current frame.
    """
    B, T, C, H, W = clip.shape
    # Clone to avoid in-place modifications
    analyzer_clip = clip.clone()
    analyzer_clip[:, -1] = recon  # replace last frame with reconstruction
    # (B, T, C, H, W) → (B, C, T, H, W)  — permute for 3-D conv backbone
    return analyzer_clip.permute(0, 2, 1, 3, 4)


# ======================================================================== #
#                          Training Loop                                   #
# ======================================================================== #

def train_one_epoch(
    model: PreprocessingSystem,
    analyzer: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 10.0,
    lam: float = 0.001,
) -> dict:
    """Run one epoch of training.

    Loss computation (inline comments map to the paper's notation):
        L = α · (L_D  +  λ · L_R)  +  L_Acc
        ─── distortion ──   ─ rate ─    ─ task ─
    """
    model.train()
    # STRICT: keep analyzer frozen + eval even inside training loop
    analyzer.eval()

    mse_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_ld = 0.0
    total_lr = 0.0
    total_lacc = 0.0
    correct = 0
    total = 0

    for batch_idx, (clip, labels) in enumerate(loader):
        # clip: (B, T, C, H, W),  labels: (B,)
        clip = clip.to(device)
        labels = labels.to(device)

        # ---- Forward through Preprocessing System ----
        enhanced, recon, rate = model(clip)  # enhanced, recon: (B,3,H,W); rate: scalar

        # Current frame = last frame of the original clip (ground truth)
        original_frame = clip[:, -1]  # (B, C, H, W)

        # ── L_D: distortion loss (MSE between original and reconstructed) ──
        loss_distortion = mse_fn(recon, original_frame)

        # ── L_R: rate loss (estimated bits-per-pixel) ──
        loss_rate = rate

        # ── L_Acc: downstream task loss ──
        # Build temporal input for the 3-D analyzer backbone
        analyzer_input = prepare_clip_for_analyzer(recon, clip)
        with torch.no_grad():
            # No gradients through the analyzer (weights frozen),
            # but we still need the computational graph through `recon`
            # which is part of analyzer_input.  Gradients will flow
            # through `recon` via the clip replacement, NOT through
            # the analyzer weights.
            pass

        # We need gradients to flow through `recon` into the preprocessor.
        # Even though analyzer weights are frozen, the *input* to the
        # analyzer (which contains `recon`) still carries grad.
        logits = analyzer(analyzer_input)  # (B, num_classes)
        loss_accuracy = ce_fn(logits, labels)

        # ── Joint loss (paper Eq.) ──
        # L = α · (L_D + λ · L_R) + L_Acc
        loss = alpha * (loss_distortion + lam * loss_rate) + loss_accuracy

        # ---- Backward + Step ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Logging ----
        total_loss += loss.item()
        total_ld += loss_distortion.item()
        total_lr += loss_rate.item()
        total_lacc += loss_accuracy.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    n = len(loader)
    return {
        "loss": total_loss / n,
        "L_D": total_ld / n,
        "L_R": total_lr / n,
        "L_Acc": total_lacc / n,
        "accuracy": correct / max(total, 1),
    }


# ======================================================================== #
#                         Validation Loop                                  #
# ======================================================================== #

@torch.no_grad()
def validate(
    model: PreprocessingSystem,
    analyzer: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float = 10.0,
    lam: float = 0.001,
) -> dict:
    """Evaluate on the mini-test split (no weight updates)."""
    model.eval()
    analyzer.eval()

    mse_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_ld = 0.0
    total_lr = 0.0
    total_lacc = 0.0
    correct = 0
    total = 0

    for clip, labels in loader:
        clip = clip.to(device)
        labels = labels.to(device)

        enhanced, recon, rate = model(clip, fq=40.0)  # fixed fq for eval
        original_frame = clip[:, -1]

        loss_distortion = mse_fn(recon, original_frame)
        loss_rate = rate

        analyzer_input = prepare_clip_for_analyzer(recon, clip)
        logits = analyzer(analyzer_input)
        loss_accuracy = ce_fn(logits, labels)

        loss = alpha * (loss_distortion + lam * loss_rate) + loss_accuracy

        total_loss += loss.item()
        total_ld += loss_distortion.item()
        total_lr += loss_rate.item()
        total_lacc += loss_accuracy.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "L_D": total_ld / n,
        "L_R": total_lr / n,
        "L_Acc": total_lacc / n,
        "accuracy": correct / max(total, 1),
    }


# ======================================================================== #
#                         Early Stopping                                   #
# ======================================================================== #

class EarlyStopping:
    """Stop training when L_Acc on validation has not improved for
    *patience* epochs.

    Tracks the best model weights so far and restores them when triggered.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.triggered = False

    def __call__(
        self, val_lacc: float, model: nn.Module
    ) -> bool:
        """Returns True if training should stop."""
        if self.best_loss is None or val_lacc < self.best_loss - self.min_delta:
            self.best_loss = val_lacc
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False

    def restore_best(self, model: nn.Module) -> None:
        """Load the best weights seen during training."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ======================================================================== #
#                              Main                                        #
# ======================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Train the Preprocessing System for Video Machine Vision"
    )
    # ---- Dataset mode ----
    parser.add_argument(
        "--dataset-mode",
        type=str,
        choices=["kinetics400", "got10k"],
        default="kinetics400",
        help="Dataset source: 'kinetics400' (HuggingFace) or 'got10k' (local folders)",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Training data directory (for got10k mode: path to val/ folder)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Test data directory (for got10k mode: path to test/ folder)",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="validation",
        help="HuggingFace split name for kinetics400 mode (default: 'validation')",
    )
    # ---- Training hyperparameters ----
    parser.add_argument("--num-frames", type=int, default=8,
                        help="Temporal depth (consecutive frames per clip)")
    parser.add_argument("--frame-stride", type=int, default=2,
                        help="Stride between sampled frames")
    parser.add_argument("--frame-size", type=int, default=224,
                        help="Spatial resolution (H=W)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam learning rate")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Weight for distortion + rate term")
    parser.add_argument("--lam", type=float, default=0.001,
                        help="Weight for rate loss within the codec term")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--analyzer-weights", type=str, default=None,
                        help="Optional path to custom analyzer checkpoint")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---- Dataset & DataLoaders ----
    frame_size = (args.frame_size, args.frame_size)

    if args.dataset_mode == "kinetics400":
        # Load from HuggingFace datasets
        print(f"[INFO] Mode: kinetics400 (HuggingFace, split='{args.hf_split}')")
        train_dataset, test_dataset, num_classes = build_kinetics400_splits(
            split=args.hf_split,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            frame_size=frame_size,
            train_ratio=0.8,
            seed=args.seed,
        )

    elif args.dataset_mode == "got10k":
        # Train on val/ folder, test on test/ folder
        if args.train_dir is None or args.test_dir is None:
            parser.error("--train-dir and --test-dir are required for got10k mode")
        print(f"[INFO] Mode: got10k")
        print(f"[INFO]   Train dir: {args.train_dir}")
        print(f"[INFO]   Test dir : {args.test_dir}")
        train_dataset, test_dataset, num_classes = build_got10k_train_test(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            frame_size=frame_size,
        )

    print(f"[INFO] Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"[INFO] Detected {num_classes} classes")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Models ----
    model = PreprocessingSystem(
        num_frames=args.num_frames,
        base_channels=64,
        latent_channels=48,
    ).to(device)

    analyzer = load_analyzer(
        num_classes=num_classes,
        device=device,
        weights_path=args.analyzer_weights,
    )

    # Verify analyzer is frozen
    trainable_analyzer = sum(p.requires_grad for p in analyzer.parameters())
    assert trainable_analyzer == 0, "Analyzer must have 0 trainable parameters!"
    print(f"[INFO] Analyzer loaded and frozen (0/{sum(1 for _ in analyzer.parameters())} params trainable)")

    # ---- Optimizer: ONLY preprocessor + codec weights ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[INFO] Adam optimizer — lr={args.lr}, "
          f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- Early Stopping ----
    early_stop = EarlyStopping(patience=args.patience)

    # ---- Checkpoint directory ----
    os.makedirs(args.save_dir, exist_ok=True)

    # ================================================================== #
    #                        Training Loop                                #
    # ================================================================== #
    print("\n" + "=" * 72)
    print("  Training Preprocessing System")
    print(f"  Loss = α·(L_D + λ·L_R) + L_Acc   |  α={args.alpha}, λ={args.lam}")
    print("=" * 72 + "\n")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- Train ----
        train_metrics = train_one_epoch(
            model, analyzer, train_loader, optimizer, device,
            alpha=args.alpha, lam=args.lam,
        )

        # ---- Validate ----
        val_metrics = validate(
            model, analyzer, test_loader, device,
            alpha=args.alpha, lam=args.lam,
        )

        elapsed = time.time() - t0

        # ---- Print epoch summary ----
        print(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s)  │  "
            f"Train  L={train_metrics['loss']:.4f}  L_D={train_metrics['L_D']:.4f}  "
            f"L_R={train_metrics['L_R']:.2f}  L_Acc={train_metrics['L_Acc']:.4f}  "
            f"Acc={train_metrics['accuracy']:.2%}  │  "
            f"Val  L={val_metrics['loss']:.4f}  L_D={val_metrics['L_D']:.4f}  "
            f"L_R={val_metrics['L_R']:.2f}  L_Acc={val_metrics['L_Acc']:.4f}  "
            f"Acc={val_metrics['accuracy']:.2%}"
        )

        # ---- Save best checkpoint ----
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}")

        # ---- Early Stopping (monitoring L_Acc on validation) ----
        if early_stop(val_metrics["L_Acc"], model):
            print(f"\n[EARLY STOP] No improvement in L_Acc for {args.patience} epochs.")
            early_stop.restore_best(model)
            print("[EARLY STOP] Restored best model weights.")
            break

    # ---- Final checkpoint ----
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)
    print(f"\n[DONE] Final model saved → {final_path}")
    print(f"[DONE] Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
