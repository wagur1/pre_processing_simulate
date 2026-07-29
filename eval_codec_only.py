"""
eval_codec_only.py — Compare Virtual Codec vs H.264 (Rate-Distortion)
========================================================================
This script evaluates ONLY the VirtualCodec's compression efficiency 
(BPP vs PSNR) and compares it against H.264 via FFmpeg. 
It does NOT use the Preprocessor or the Analyzer.
"""

import argparse
import os
import subprocess
import tempfile
import math

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_kaggle_kinetics400_splits
from model import VirtualCodec

def compute_psnr(mse):
    if mse == 0:
        return 100.0
    # max pixel value is 1.0 since tensors are in [0, 1]
    return 10 * math.log10(1.0 / mse)

def run_ffmpeg(input_path: str, output_path: str, qp: int) -> int:
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
    return os.path.getsize(output_path)

def compress_and_decode_h264(frame_tensor: torch.Tensor, qp: int, temp_dir: str):
    """Compress frame via H.264, decode, and return (decoded_tensor, bpp)"""
    frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    in_img = os.path.join(temp_dir, "input.png")
    out_vid = os.path.join(temp_dir, "out.mp4")
    
    cv2.imwrite(in_img, frame_bgr)
    
    file_size_bytes = run_ffmpeg(in_img, out_vid, qp)
    _, H, W = frame_tensor.shape
    bpp = (file_size_bytes * 8) / (H * W)
    
    cap = cv2.VideoCapture(out_vid)
    ret, dec_bgr = cap.read()
    cap.release()
    
    if not ret:
        dec_bgr = frame_bgr
        
    dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
    dec_tensor = torch.from_numpy(dec_rgb).permute(2, 0, 1).float() / 255.0
    return dec_tensor.to(frame_tensor.device), bpp

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, required=True, help="Path to Kinetics400 test videos")
    parser.add_argument("--codec-type", type=str, choices=["virtual", "compressai"], default="virtual")
    parser.add_argument("--codec-weights", type=str, default=None, help="Path to codec_pretrained.pt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    _, test_ds, _ = build_kaggle_kinetics400_splits(
        os.path.dirname(args.test_dir),
        args.test_dir,
        num_frames=16
    )
    
    if args.limit:
        test_ds.samples = test_ds.samples[:args.limit]
        test_ds.labels = test_ds.labels[:args.limit]
        
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.codec_type == "compressai":
        from model import CompressAICodec
        codec = CompressAICodec(quality=3).to(device)
    else:
        codec = VirtualCodec(latent_channels=48).to(device)
        if args.codec_weights:
            ckpt = torch.load(args.codec_weights, map_location=device)
            if "codec_state_dict" in ckpt:
                codec.load_state_dict(ckpt["codec_state_dict"])
            else:
                codec.load_state_dict(ckpt)
    codec.eval()
    
    qps = [30, 35, 40, 45, 50]
    # For Virtual Codec, we use equivalent fq values
    fqs = [30.0, 35.0, 40.0, 45.0, 50.0]
    
    results = {
        "h264": {qp: {"bpp": 0.0, "mse": 0.0, "count": 0} for qp in qps},
        "virtual": {fq: {"bpp": 0.0, "mse": 0.0, "count": 0} for fq in fqs},
    }
    
    mse_fn = nn.MSELoss()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for batch_idx, (clip, _) in enumerate(tqdm(loader, desc="Eval Codecs")):
            clip = clip.to(device)
            B = clip.shape[0]
            original_frames = clip[:, 7]
            
            for b_idx in range(B):
                orig_f = original_frames[b_idx]
                
                # --- H.264 ---
                for qp in qps:
                    dec_h264, bpp_h264 = compress_and_decode_h264(orig_f, qp, temp_dir)
                    mse_h264 = mse_fn(dec_h264, orig_f).item()
                    
                    results["h264"][qp]["bpp"] += bpp_h264
                    results["h264"][qp]["mse"] += mse_h264
                    results["h264"][qp]["count"] += 1
                    
                # --- Virtual Codec ---
                for fq in fqs:
                    # Virtual Codec expects batched input (1, C, H, W)
                    orig_f_batched = orig_f.unsqueeze(0)
                    dec_virt, rate_virt = codec(orig_f_batched, fq=fq)
                    
                    mse_virt = mse_fn(dec_virt.squeeze(0), orig_f).item()
                    bpp_virt = rate_virt.item()
                    
                    results["virtual"][fq]["bpp"] += bpp_virt
                    results["virtual"][fq]["mse"] += mse_virt
                    results["virtual"][fq]["count"] += 1

    print("\n=============================================================")
    print("                RATE-DISTORTION COMPARISON                   ")
    print("=============================================================")
    print("H.264 (x264, preset medium):")
    print(f"{'QP':<5} | {'BPP':<15} | {'MSE':<15} | {'PSNR (dB)':<15}")
    for qp in qps:
        cnt = max(1, results["h264"][qp]["count"])
        bpp = results["h264"][qp]["bpp"] / cnt
        mse = results["h264"][qp]["mse"] / cnt
        psnr = compute_psnr(mse)
        print(f"{qp:<5} | {bpp:<15.4f} | {mse:<15.4f} | {psnr:<15.2f}")
        
    print("\nVirtual Codec (Neural Compression):")
    print(f"{'fq':<5} | {'BPP':<15} | {'MSE':<15} | {'PSNR (dB)':<15}")
    for fq in fqs:
        cnt = max(1, results["virtual"][fq]["count"])
        bpp = results["virtual"][fq]["bpp"] / cnt
        mse = results["virtual"][fq]["mse"] / cnt
        psnr = compute_psnr(mse)
        print(f"{fq:<5} | {bpp:<15.4f} | {mse:<15.4f} | {psnr:<15.2f}")
    print("=============================================================")

if __name__ == "__main__":
    main()
