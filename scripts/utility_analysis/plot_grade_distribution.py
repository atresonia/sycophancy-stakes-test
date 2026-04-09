"""
Grouped bar chart: % of essays awarded each letter grade for Baseline / Low / High stakes.

Usage:
    python scripts/utility_analysis/plot_grade_distribution.py \
        --in_json data/essay_grading/it-variants/llm_resp_stakes_gemini_variants.jsonl \
        --out_png  plots/grade_distribution.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from utils.utility import GRADES, LOW_KEYS, HIGH_KEYS, majority, load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", default="data/essay_grading/it-variants/llm_resp_stakes_gemini_variants.jsonl")
    parser.add_argument("--out_png", default="plots/grade_distribution.png")
    args = parser.parse_args()

    records = load_jsonl(Path(args.in_json))
    n = len(records)
    print(f"Loaded {n} essays")

    baseline = [r["baseline_output"]["grade"] for r in records]
    low = [majority([r[f"{k}_output"]["grade"] for k in LOW_KEYS]) for r in records]
    high = [majority([r[f"{k}_output"]["grade"] for k in HIGH_KEYS]) for r in records]

    conditions = {"Baseline": baseline, "Low Stakes": low, "High Stakes": high}
    colors = {"Baseline": "#888888", "Low Stakes": "#5b9bd5", "High Stakes": "#c0392b"}

    x = np.arange(len(GRADES))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(7, 4))
    for (label, grades), offset in zip(conditions.items(), offsets):
        counts = Counter(grades)
        pcts = [counts.get(g, 0) / n * 100 for g in GRADES]
        ax.bar(x + offset, pcts, width, label=label, color=colors[label])

    ax.set_xticks(x)
    ax.set_xticklabels(GRADES, fontsize=12)
    ax.set_ylabel("% of essays", fontsize=11)
    ax.set_xlabel("Grade awarded", fontsize=11)
    ax.set_title("Grade Distribution by Stakes Level", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 70)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
