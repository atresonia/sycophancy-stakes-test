"""
Histogram of grade shift from baseline, broken down by true grade class.
Shift = majority_vote(stakes framings) − baseline grade (in grade steps).

One panel per true grade class (B, C, D, F if present).
Two grouped bars per shift bucket: Low Stakes vs High Stakes.

Usage:
    python scripts/utility_analysis/plot_grade_shift_histogram.py \
        --in_json data/essay_grading/it-variants/llm_resp_stakes_gemini_variants.jsonl \
        --out_png  plots/grade_shift_by_class_gemini.png
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRADE_NUM = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
LOW_KEYS  = ["low_stakes_1",  "low_stakes_2",  "low_stakes_3"]
HIGH_KEYS = ["high_stakes_1", "high_stakes_2", "high_stakes_3"]
CONDITIONS = [
    ("Low Stakes",  LOW_KEYS,  "#5b9bd5"),
    ("High Stakes", HIGH_KEYS, "#c0392b"),
]
SHIFT_RANGE = [-2, -1, 0, 1]


def majority(grades: list[str]) -> str:
    return Counter(grades).most_common(1)[0][0]


def load(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("status") == "ok":
                    records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", default="data/essay_grading/it-variants/llm_resp_stakes_gemini_variants.jsonl")
    parser.add_argument("--out_png",  default="plots/grade_shift_by_class.png")
    args = parser.parse_args()

    records = load(Path(args.in_json))
    print(f"Loaded {len(records)} essays\n")

    by_grade: dict[str, list[dict]] = {}
    for r in records:
        by_grade.setdefault(r["ground_truth"], []).append(r)
    true_grades = [g for g in ["B", "C", "D", "F"] if g in by_grade]

    shifts: dict[str, dict[str, list[int]]] = {g: {} for g in true_grades}
    for label, keys, _ in CONDITIONS:
        for g in true_grades:
            shifts[g][label] = [
                GRADE_NUM[majority([r[f"{k}_output"]["grade"] for k in keys])]
                - GRADE_NUM[r["baseline_output"]["grade"]]
                for r in by_grade[g]
            ]

    n_panels = len(true_grades)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4), sharey=False)
    if n_panels == 1:
        axes = [axes]

    x = np.arange(len(SHIFT_RANGE))
    bar_width = 0.38
    tick_labels = [str(s) if s < 0 else f"+{s}" if s > 0 else "0" for s in SHIFT_RANGE]

    for ax, g in zip(axes, true_grades):
        n = len(shifts[g][CONDITIONS[0][0]])
        for i, (label, _, color) in enumerate(CONDITIONS):
            bucket = Counter(shifts[g][label])
            counts = [bucket[s] for s in SHIFT_RANGE]
            offset = (i - 0.5) * bar_width
            ax.bar(x + offset, counts, bar_width, label=label, color=color)

        ax.set_title(f"True grade: {g}\n(n={n})", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Grade steps from baseline", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Number of essays", fontsize=10)
    fig.suptitle("Grade Shift from Baseline by True Grade", fontsize=13, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
