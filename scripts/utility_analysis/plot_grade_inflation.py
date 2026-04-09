"""
Grade shift from baseline: number of essays that graded stricter, same, or more lenient
under Low Stakes vs High Stakes. One panel per condition.

Usage:
    python scripts/utility_analysis/plot_grade_inflation.py \
        --in_json data/essay_grading/it-variants/llm_resp_stakes_gemini_variants.jsonl \
        --out_png  plots/grade_shift.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from collections import Counter

from utils.utility import GRADE_NUM, LOW_KEYS, HIGH_KEYS, majority, load_jsonl


def shift_color(delta: int) -> str:
    if delta < 0:
        return "#c0392b"
    if delta == 0:
        return "#888888"
    return "#27ae60"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", default="data/essay_grading/it-variants/llm_resp_stakes_gemini_variants.jsonl")
    parser.add_argument("--out_png", default="plots/grade_shift.png")
    args = parser.parse_args()

    records = load_jsonl(Path(args.in_json))
    print(f"Loaded {len(records)} essays")

    conditions = [("Low Stakes", LOW_KEYS), ("High Stakes", HIGH_KEYS)]
    all_shifts: dict[str, list[int]] = {}
    for label, keys in conditions:
        all_shifts[label] = [
            GRADE_NUM[majority([r[f"{k}_output"]["grade"] for k in keys])]
            - GRADE_NUM[r["baseline_output"]["grade"]]
            for r in records
        ]

    all_vals = [d for shifts in all_shifts.values() for d in shifts]
    shift_range = list(range(min(all_vals), max(all_vals) + 1))
    x = np.arange(len(shift_range))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for ax, (label, _) in zip(axes, conditions):
        counts = Counter(all_shifts[label])
        heights = [counts.get(d, 0) for d in shift_range]
        colors = [shift_color(d) for d in shift_range]
        ax.bar(x, heights, width=0.6, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:+d}" if d != 0 else "0" for d in shift_range], fontsize=11)
        ax.set_xlabel("Grade steps from baseline", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Number of essays", fontsize=11)
    fig.suptitle("Grade Shift from Baseline", fontsize=13, fontweight="bold", y=1.02)
    fig.legend(
        handles=[Patch(color=c, label=l) for c, l in [
            ("#c0392b", "Stricter"), ("#888888", "Same"), ("#27ae60", "More lenient")
        ]],
        loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
