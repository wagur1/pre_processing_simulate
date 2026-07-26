"""
visualize.py — Visualize Original vs Compressed Frames
======================================================
This script extracts a random frame from the test set and visualizes:
1. Original Frame
2. H.264 Decoded Frame (Baseline)
3. Virtual Codec Decoded Frame
4. Preprocessor + H.264 Decoded Frame (Proposed)

It saves the comparison as a PNG image for easy viewing.
"""

import argparse
import os
import subprocess
import tempfile
import random

import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import build_kaggle_kinetics400_splits
from model import VirtualCodec, Preprocessor

def run_ffmpeg(input_path: str, output_path: str, qp: int):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(qp),
        "-frames:v", "1",
        output_path
    ]
    subprocess.run(cmd, check=True)

def compress_and_decode_h264(frame_tensor: torch.Tensor, qp: int, temp_dir: str):
    """Compress frame via H.264, decode, and return decoded tensor"""
    frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    in_img = os.path.join(temp_dir, "input.png")
    out_vid = os.path.join(temp_dir, "out.mp4")
    
    cv2.imwrite(in_img, frame_bgr)
    run_ffmpeg(in_img, out_vid, qp)
    
    cap = cv2.VideoCapture(out_vid)
    ret, dec_bgr = cap.read()
    cap.release()
    
    if not ret: dec_bgr = frame_bgr
        
    dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
    dec_tensor = torch.from_numpy(dec_rgb).permute(2, 0, 1).float() / 255.0
    return dec_tensor.to(frame_tensor.device)

def tensor_to_plot_img(tensor):
    # Convert (C, H, W) [0, 1] tensor to (H, W, C) [0, 1] numpy array for matplotlib
    return tensor.permute(1, 2, 0).cpu().numpy().clip(0, 1)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--codec-weights", type=str, required=True)
    parser.add_argument("--preprocessor-weights", type=str, required=True)
    parser.add_argument("--qp", type=int, default=40, help="Quality parameter for H.264")
    parser.add_argument("--output", type=str, default="visualization.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load dataset
    _, test_ds, _ = build_kaggle_kinetics400_splits(os.path.dirname(args.test_dir), args.test_dir)
    
    # Pick a random sample
    idx = random.randint(0, len(test_ds) - 1)
    clip, label = test_ds[idx]
    clip = clip.unsqueeze(0).to(device)  # Add batch dim (1, T, C, H, W)
    
    # Load Models
    codec = VirtualCodec(latent_channels=48).to(device)
    codec.load_state_dict(torch.load(args.codec_weights, map_location=device)["codec_state_dict"])
    codec.eval()

    preprocessor = Preprocessor(num_frames=3, base_channels=64).to(device)
    ckpt = torch.load(args.preprocessor_weights, map_location=device)
    if "model_state_dict" in ckpt:
        prep_state = {k.replace("preprocessor.", ""): v for k, v in ckpt["model_state_dict"].items() if k.startswith("preprocessor.")}
        preprocessor.load_state_dict(prep_state)
    else:
        preprocessor.load_state_dict(ckpt)
    preprocessor.eval()

    # Process Images
    original_frame = clip[0, 1]  # Middle frame
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. H.264 Baseline
        dec_h264 = compress_and_decode_h264(original_frame, args.qp, temp_dir)
        
        # 2. Virtual Codec
        dec_virt, _ = codec(original_frame.unsqueeze(0), fq=float(args.qp))
        dec_virt = dec_virt.squeeze(0)
        
        # 3. Proposed (Preprocessor + H.264)
        enhanced_frame = preprocessor(clip)[0]
        dec_proposed = compress_and_decode_h264(enhanced_frame, args.qp, temp_dir)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Image Compression Visualization (QP/fq = {args.qp})", fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(tensor_to_plot_img(original_frame))
    axes[0, 0].set_title("1. Original Frame", fontsize=12)
    axes[0, 0].axis('off')

    # Baseline H.264
    axes[0, 1].imshow(tensor_to_plot_img(dec_h264))
    axes[0, 1].set_title("2. Baseline H.264", fontsize=12)
    axes[0, 1].axis('off')

    # Virtual Codec
    axes[1, 0].imshow(tensor_to_plot_img(dec_virt))
    axes[1, 0].set_title("3. Virtual Codec Decoded", fontsize=12)
    axes[1, 0].axis('off')

    # Proposed
    axes[1, 1].imshow(tensor_to_plot_img(dec_proposed))
    axes[1, 1].set_title("4. Proposed (Preprocessor + H.264)", fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n[DONE] Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
