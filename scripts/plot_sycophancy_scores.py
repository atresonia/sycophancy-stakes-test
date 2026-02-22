"""
Script to plot low stakes vs high stakes sycophancy scores.

Usage:
    python scripts/plot_sycophancy_scores.py --in_json data/aita-yta-openai-eval.jsonl --out_png plots/sycophancy_scores.png
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_scores(jsonl_path: Path) -> tuple[list[int], list[int], list[int]]:
    """Load sycophancy scores from evaluation jsonl file."""
    prompt_ids = []
    low_scores = []
    high_scores = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok":
                continue
            
            output = record.get("output", {})
            prompt_ids.append(record.get("prompt_row_id", 0))
            low_scores.append(output.get("low_stakes_score", 0))
            high_scores.append(output.get("high_stakes_score", 0))
    
    return prompt_ids, low_scores, high_scores


def plot_scores(prompt_ids: list[int], low_scores: list[int], high_scores: list[int], out_path: Path):
    """Create scatter plot of low vs high stakes sycophancy scores."""
    
    # Set up the plot with a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color palette
    primary_color = '#2E86AB'  # Steel blue
    accent_color = '#A23B72'   # Magenta pink
    
    # Create scatter plot
    scatter = ax.scatter(
        low_scores, 
        high_scores, 
        c=primary_color,
        s=120,
        alpha=0.7,
        edgecolors='white',
        linewidths=1.5,
        zorder=3
    )
    
    # Add diagonal reference line (y = x)
    max_val = max(max(low_scores), max(high_scores), 50)
    ax.plot([0, max_val], [0, max_val], 
            color=accent_color, 
            linestyle='--', 
            linewidth=2, 
            alpha=0.7, 
            label='Equal sycophancy (y=x)',
            zorder=2)
    
    # Add labels for each point
    for i, pid in enumerate(prompt_ids):
        ax.annotate(
            str(pid),
            (low_scores[i], high_scores[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            color='#333333',
            alpha=0.8
        )
    
    # Styling
    ax.set_xlabel('Low Stakes Sycophancy Score', fontsize=14, fontweight='medium', color='#333333')
    ax.set_ylabel('High Stakes Sycophancy Score', fontsize=14, fontweight='medium', color='#333333')
    ax.set_title('Sycophancy: Low Stakes vs High Stakes', fontsize=18, fontweight='bold', color='#1a1a1a', pad=20)
    
    # Set axis limits with padding
    padding = 5
    ax.set_xlim(-padding, max_val + padding)
    ax.set_ylim(-padding, max_val + padding)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add summary statistics
    mean_low = np.mean(low_scores)
    mean_high = np.mean(high_scores)
    diff = mean_high - mean_low
    
    stats_text = (
        f"n = {len(prompt_ids)}\n"
        f"Mean Low: {mean_low:.1f}\n"
        f"Mean High: {mean_high:.1f}\n"
        f"Δ (High - Low): {diff:+.1f}"
    )
    
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9),
        fontfamily='monospace'
    )
    
    # Shade regions
    # Above diagonal = higher sycophancy in high stakes
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val], 
                    alpha=0.05, color=accent_color, zorder=1)
    ax.text(max_val * 0.15, max_val * 0.85, 'Higher sycophancy\nin high stakes', 
            fontsize=10, color=accent_color, alpha=0.7, style='italic')
    
    # Below diagonal = higher sycophancy in low stakes  
    ax.fill_between([0, max_val], [0, 0], [0, max_val],
                    alpha=0.05, color=primary_color, zorder=1)
    ax.text(max_val * 0.65, max_val * 0.15, 'Higher sycophancy\nin low stakes',
            fontsize=10, color=primary_color, alpha=0.7, style='italic')
    
    plt.tight_layout()
    
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {out_path}")
    
    # Also show
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot low vs high stakes sycophancy scores")
    parser.add_argument("--in_json", type=str, default="data/aita-yta-openai-eval.jsonl",
                        help="Input JSONL file with evaluation results")
    parser.add_argument("--out_png", type=str, default="plots/sycophancy_scores.png",
                        help="Output PNG file path")
    args = parser.parse_args()
    
    in_path = Path(args.in_json)
    out_path = Path(args.out_png)
    
    if not in_path.exists():
        print(f"Error: Input file not found: {in_path}")
        return
    
    prompt_ids, low_scores, high_scores = load_scores(in_path)
    print(f"Loaded {len(prompt_ids)} evaluation records")
    
    if not prompt_ids:
        print("No valid records found!")
        return
    
    plot_scores(prompt_ids, low_scores, high_scores, out_path)


if __name__ == "__main__":
    main()
