"""
train_codec.py — Pre-training the Virtual Codec
=================================================
This script trains ONLY the VirtualCodec component on image reconstruction.
It is required because initializing the VirtualCodec with random weights and
freezing it would result in random noise compression, making it impossible
for the Preprocessor to learn anything meaningful.

Loss: L_D + λ * L_R
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    build_kinetics400_splits,
    build_got10k_train_test,
    build_kaggle_kinetics400_splits,
    build_kaggle_got10k_train_test,
)
from model import VirtualCodec
from kaggle_config import is_kaggle, get_kaggle_save_dir

def train_one_epoch(
    model: VirtualCodec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lam: float = 0.001,
) -> dict:
    model.train()
    mse_fn = nn.MSELoss()
    
    total_loss = 0.0
    total_ld = 0.0
    total_lr = 0.0

    for batch_idx, (clip, _) in enumerate(loader):
        clip = clip.to(device)
        # Use the middle frame for reconstruction (T=3)
        original_frame = clip[:, 1]
        
        # Forward through VirtualCodec
        recon, rate = model(original_frame)
        
        loss_distortion = mse_fn(recon, original_frame)
        loss_rate = rate
        
        loss = loss_distortion + lam * loss_rate
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ld += loss_distortion.item()
        total_lr += loss_rate.item()
        
    n = len(loader)
    return {
        "loss": total_loss / n,
        "L_D": total_ld / n,
        "L_R": total_lr / n,
    }

@torch.no_grad()
def validate(
    model: VirtualCodec,
    loader: DataLoader,
    device: torch.device,
    lam: float = 0.001,
) -> dict:
    model.eval()
    mse_fn = nn.MSELoss()
    
    total_loss = 0.0
    total_ld = 0.0
    total_lr = 0.0

    for clip, _ in loader:
        clip = clip.to(device)
        original_frame = clip[:, 1]
        
        # Fixed fq=40.0 for validation
        recon, rate = model(original_frame, fq=40.0)
        
        loss_distortion = mse_fn(recon, original_frame)
        loss_rate = rate
        loss = loss_distortion + lam * loss_rate
        
        total_loss += loss.item()
        total_ld += loss_distortion.item()
        total_lr += loss_rate.item()
        
    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "L_D": total_ld / n,
        "L_R": total_lr / n,
    }

def main():
    parser = argparse.ArgumentParser(description="Pre-train Virtual Codec")
    parser.add_argument("--dataset-mode", type=str, default="kaggle_kinetics400")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--kaggle", action="store_true")
    args = parser.parse_args()

    if args.kaggle or is_kaggle():
        args.save_dir = get_kaggle_save_dir()
        args.train_dir = "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train"
        args.test_dir = "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/test"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # For dataset, we use num_frames=3 since that's what the dataset returns by default now
    if args.dataset_mode == "kaggle_kinetics400":
        train_ds, test_ds, _ = build_kaggle_kinetics400_splits(args.train_dir, args.test_dir)
    else:
        raise NotImplementedError("Only kaggle_kinetics400 supported in this minimal script for now.")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VirtualCodec(latent_channels=48).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    print("\n" + "=" * 50)
    print("  Pre-training Virtual Codec")
    print(f"  Loss = L_D + {args.lam} * L_R")
    print("=" * 50 + "\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, lam=args.lam)
        val_metrics = validate(model, test_loader, device, lam=args.lam)
        
        elapsed = time.time() - t0
        
        print(
            f"Epoch {epoch:2d}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train L={train_metrics['loss']:.4f} L_D={train_metrics['L_D']:.4f} L_R={train_metrics['L_R']:.2f} | "
            f"Val L={val_metrics['loss']:.4f} L_D={val_metrics['L_D']:.4f} L_R={val_metrics['L_R']:.2f}"
        )
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = os.path.join(args.save_dir, "codec_pretrained.pt")
            torch.save({
                "epoch": epoch,
                "codec_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  ✓ Saved best codec checkpoint → {ckpt_path}")

if __name__ == "__main__":
    main()
