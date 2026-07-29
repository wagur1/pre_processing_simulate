import matplotlib.pyplot as plt

# Data from Kaggle output
baseline_bpp = [0.2893, 0.3051, 0.3392, 0.3904, 0.4856]
baseline_acc = [1.68, 1.64, 1.77, 2.07, 2.33]

proposed_bpp = [0.4038, 0.5192, 1.4421, 1.7107, 2.1074]
proposed_acc = [2.76, 2.50, 1.94, 2.15, 2.15]

plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(baseline_bpp, baseline_acc, marker='o', label='Baseline H.264', linestyle='-', linewidth=2, markersize=8)
plt.plot(proposed_bpp, proposed_acc, marker='s', label='Proposed + H.264', linestyle='-', linewidth=2, markersize=8)

# Formatting
plt.title('Rate-Accuracy Performance (H.264)', fontsize=14, fontweight='bold')
plt.xlabel('Bitrate (BPP)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Save the plot
output_path = r'C:\Users\Wagur1\.gemini\antigravity\brain\ef319dd0-78ab-4281-95cb-541836f91915\h264_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {output_path}")
