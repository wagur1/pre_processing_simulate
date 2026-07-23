"""
eval_real_codec.py — Real Codec Testing (H.264 / H.265)
======================================================
This script evaluates the trained Preprocessor against standard H.264 and
H.265 video codecs, replicating the experimental setup of the paper:
"A Preprocessing Framework for Video Machine Vision under Compression".

Workflow per test clip:
1. Baseline: Compress original frame (T=3 middle frame) -> Decode -> Measure Accuracy & BPP.
2. Proposed: Preprocess clip -> Compress enhanced frame -> Decode -> Measure Accuracy & BPP.
"""

import argparse
import os
import time
import subprocess
import tempfile
from typing import Tuple

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_kaggle_kinetics400_splits
from model import Preprocessor
from train import load_analyzer
from kaggle_config import is_kaggle

def run_ffmpeg(input_path: str, output_path: str, codec: str, qp: int) -> int:
    """Run ffmpeg to compress a single image/frame into a video stream.
    Returns the size of the compressed bitstream in bytes.
    """
    if codec == "h264":
        encoder = "libx264"
        crf_flag = "-crf"
    elif codec == "h265":
        encoder = "libx265"
        crf_flag = "-crf"
    else:
        raise ValueError(f"Unknown codec: {codec}")

    # Use medium preset as specified in the paper
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_path,
        "-c:v", encoder,
        "-preset", "medium",
        crf_flag, str(qp),
        # Ensure it treats it as a 1-frame video
        "-frames:v", "1",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return os.path.getsize(output_path)

def compress_and_decode(frame_tensor: torch.Tensor, codec: str, qp: int, temp_dir: str) -> Tuple[torch.Tensor, float]:
    """Compress a single frame tensor via ffmpeg, decode it back, and compute BPP."""
    # frame_tensor: (C, H, W) in [0, 1] RGB
    frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    in_img = os.path.join(temp_dir, "input.png")
    out_vid = os.path.join(temp_dir, f"out.{'mp4' if codec == 'h264' else 'mkv'}")
    
    cv2.imwrite(in_img, frame_bgr)
    
    # Compress
    file_size_bytes = run_ffmpeg(in_img, out_vid, codec, qp)
    
    # BPP = (bytes * 8) / (H * W)
    _, H, W = frame_tensor.shape
    bpp = (file_size_bytes * 8) / (H * W)
    
    # Decode
    cap = cv2.VideoCapture(out_vid)
    ret, dec_bgr = cap.read()
    cap.release()
    
    if not ret:
        dec_bgr = frame_bgr # Fallback if decode fails
        
    dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
    dec_tensor = torch.from_numpy(dec_rgb).permute(2, 0, 1).float() / 255.0
    return dec_tensor.to(frame_tensor.device), bpp

@torch.no_grad()
def evaluate_qps(
    preprocessor: nn.Module,
    analyzer: nn.Module,
    loader: DataLoader,
    device: torch.device,
    codec: str = "h264",
    qps: list = [30, 35, 40, 45, 50],
):
    preprocessor.eval()
    analyzer.eval()
    
    results = {
        "baseline": {qp: {"bpp": 0.0, "acc": 0.0, "count": 0} for qp in qps},
        "proposed": {qp: {"bpp": 0.0, "acc": 0.0, "count": 0} for qp in qps},
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for batch_idx, (clip, labels) in enumerate(tqdm(loader, desc=f"Eval {codec}")):
            clip = clip.to(device)
            labels = labels.to(device)
            
            # Batch size must be 1 for ffmpeg wrapper simplicity, or we process frame by frame
            B, T, C, H, W = clip.shape
            
            # 1. Baseline: Compress original middle frame
            original_frames = clip[:, 1]
            
            # 2. Proposed: Preprocess first
            enhanced_frames = preprocessor(clip)
            
            for b_idx in range(B):
                orig_f = original_frames[b_idx]
                enh_f = enhanced_frames[b_idx]
                label = labels[b_idx].unsqueeze(0)
                
                for qp in qps:
                    # --- Baseline ---
                    dec_base, bpp_base = compress_and_decode(orig_f, codec, qp, temp_dir)
                    # Prepare 3D input for analyzer (B=1, C, T, H, W)
                    base_clip = clip[b_idx].clone()
                    base_clip[1] = dec_base
                    base_input = base_clip.permute(1, 0, 2, 3).unsqueeze(0)
                    
                    logits_base = analyzer(base_input)
                    pred_base = logits_base.argmax(dim=1)
                    acc_base = (pred_base == label).sum().item()
                    
                    results["baseline"][qp]["bpp"] += bpp_base
                    results["baseline"][qp]["acc"] += acc_base
                    results["baseline"][qp]["count"] += 1
                    
                    # --- Proposed ---
                    dec_enh, bpp_enh = compress_and_decode(enh_f, codec, qp, temp_dir)
                    enh_clip = clip[b_idx].clone()
                    enh_clip[1] = dec_enh
                    enh_input = enh_clip.permute(1, 0, 2, 3).unsqueeze(0)
                    
                    logits_enh = analyzer(enh_input)
                    pred_enh = logits_enh.argmax(dim=1)
                    acc_enh = (pred_enh == label).sum().item()
                    
                    results["proposed"][qp]["bpp"] += bpp_enh
                    results["proposed"][qp]["acc"] += acc_enh
                    results["proposed"][qp]["count"] += 1

    # Print summary
    print(f"\n[{codec.upper()}] Evaluation Results:")
    print("="*60)
    print(f"{'QP':<5} | {'Baseline BPP':<15} | {'Baseline Acc':<15} | {'Proposed BPP':<15} | {'Proposed Acc':<15}")
    print("-" * 60)
    for qp in qps:
        cnt = max(1, results["baseline"][qp]["count"])
        b_bpp = results["baseline"][qp]["bpp"] / cnt
        b_acc = results["baseline"][qp]["acc"] / cnt
        
        p_bpp = results["proposed"][qp]["bpp"] / cnt
        p_acc = results["proposed"][qp]["acc"] / cnt
        
        print(f"{qp:<5} | {b_bpp:<15.4f} | {b_acc:<15.2%} | {p_bpp:<15.4f} | {p_acc:<15.2%}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Preprocessor with Real Codecs")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to Kinetics400 test videos")
    parser.add_argument("--preprocessor-weights", type=str, required=True, help="Path to trained model.pt")
    parser.add_argument("--codec", type=str, choices=["h264", "h265", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load Dataset (T=3)
    _, test_ds, num_classes = build_kaggle_kinetics400_splits(
        os.path.dirname(args.test_dir), # Not perfectly generic, but works for the current split structure
        args.test_dir
    )
    
    if args.limit:
        test_ds.samples = test_ds.samples[:args.limit]
        test_ds.labels = test_ds.labels[:args.limit]
        
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load Analyzer
    analyzer = load_analyzer(num_classes=num_classes, device=device)

    # Load Preprocessor
    preprocessor = Preprocessor(num_frames=3, base_channels=64).to(device)
    ckpt = torch.load(args.preprocessor_weights, map_location=device)
    
    # We trained `PreprocessingSystem`, so we need to extract preprocessor weights
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        # Filter only preprocessor weights
        prep_state = {k.replace("preprocessor.", ""): v for k, v in state_dict.items() if k.startswith("preprocessor.")}
        preprocessor.load_state_dict(prep_state)
    else:
        preprocessor.load_state_dict(ckpt)
        
    print(f"[INFO] Loaded preprocessor weights from {args.preprocessor_weights}")

    # Run evaluations
    codecs_to_run = ["h264", "h265"] if args.codec == "both" else [args.codec]
    for c in codecs_to_run:
        evaluate_qps(preprocessor, analyzer, test_loader, device, codec=c)

if __name__ == "__main__":
    main()
