"""
Mean grade inflation vs ground truth, with 95% CI, broken down by true grade class.
Grade inflation = awarded grade − baseline LLM grade (positive = more lenient than baseline).

One panel per true grade class (B, C, D). One bar per stakes condition (Low, High).
Also prints a summary table of mean inflation per condition per class.

Usage:
    python scripts/utility_analysis/plot_grade_shift_by_class.py \
        --in_json data/essay_grading/it-variants/llm_resp_stakes_gemini_variants.jsonl \
        --out_png  plots/grade_inflation_by_class.png

output format (stdout):
True Grade  Low Stakes    High Stakes   Average
------------------------------------------------------
B           -1.286        -1.457        -1.371
C           +0.143        +0.041        +0.092
D           +0.438        +0.312        +0.375
------------------------------------------------------
Average     -0.310        -0.440        -0.375
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GRADE_NUM = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
LOW_KEYS = ["low_stakes_1", "low_stakes_2", "low_stakes_3"]
HIGH_KEYS = ["high_stakes_1", "high_stakes_2", "high_stakes_3"]
CONDITIONS = [("Low Stakes", LOW_KEYS, "#5b9bd5"), ("High Stakes", HIGH_KEYS, "#c0392b")]


def majority(grades: list[str]) -> str:
    return Counter(grades).most_common(1)[0][0]


def load(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ci95(values: list[float]) -> float:
    """Half-width of 95% confidence interval assuming normal approximation."""
    a = np.array(values)
    return 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", default="data/essay_grading/it-variants/llm_resp_stakes_gemini_variants.jsonl")
    parser.add_argument("--out_png", default=None)
    parser.add_argument("--model", default="gemini")
    args = parser.parse_args()

    records = load(Path(args.in_json))
    print(f"Loaded {len(records)} essays\n")

    true_grades = [g for g in ["B", "C", "D", "F"]
                   if any(r["ground_truth"] == g for r in records)]

    # Compute inflation (given - baseline_llm_grade) per essay per condition
    # Using majority vote across the 3 framings → one value per essay per condition
    inflation: dict[str, dict[str, list[float]]] = {g: {} for g in true_grades}
    for label, keys, _ in CONDITIONS:
        for g in true_grades:
            subset = [r for r in records if r["ground_truth"] == g
                      and r.get("baseline_output", {}).get("grade") in GRADE_NUM]
            inflation[g][label] = [
                GRADE_NUM[majority([r[f"{k}_output"]["grade"] for k in keys])]
                - GRADE_NUM[r["baseline_output"]["grade"]]
                for r in subset
            ]

    # --- Print summary table ---
    col_w = 14
    header = f"{'True Grade':<12}" + "".join(f"{l:<{col_w}}" for l, _, _ in CONDITIONS) + f"{'Average':<{col_w}}"
    print(header)
    print("-" * len(header))

    all_low, all_high = [], []
    for g in true_grades:
        row = f"{g:<12}"
        means = []
        for label, _, _ in CONDITIONS:
            vals = inflation[g][label]
            m = np.mean(vals)
            means.append(m)
            row += f"{m:+.3f}{'':<{col_w - 6}}"
            if label == "Low Stakes":
                all_low.extend(vals)
            else:
                all_high.extend(vals)
        avg = np.mean(means)
        row += f"{avg:+.3f}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'Average':<12}"
    overall_means = []
    for vals in [all_low, all_high]:
        m = np.mean(vals)
        avg_row += f"{m:+.3f}{'':<{col_w - 6}}"
        overall_means.append(m)
    avg_row += f"{np.mean(overall_means):+.3f}"
    print(avg_row)
    print()

    # --- Plot ---
    x = np.arange(len(CONDITIONS))
    width = 0.5

    fig, axes = plt.subplots(1, len(true_grades), figsize=(3.2 * len(true_grades), 4), sharey=True)
    if len(true_grades) == 1:
        axes = [axes]

    for ax, g in zip(axes, true_grades):
        n = len(inflation[g][CONDITIONS[0][0]])
        means = [np.mean(inflation[g][label]) for label, _, _ in CONDITIONS]
        errors = [ci95(inflation[g][label]) for label, _, _ in CONDITIONS]
        colors = [c for _, _, c in CONDITIONS]

        bars = ax.bar(x, means, width, color=colors, yerr=errors,
                      capsize=6, error_kw={"linewidth": 1.5, "ecolor": "#333333"})
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        for bar, mean, err in zip(bars, means, errors):
            bar_center = mean / 2
            # Place inside bar only if tall enough; otherwise above/below with offset
            if abs(mean) >= 0.15:
                ax.text(bar.get_x() + bar.get_width() / 2, bar_center,
                        f"{mean:+.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
            else:
                offset = (err + 0.08) * (1 if mean >= 0 else -1)
                ax.text(bar.get_x() + bar.get_width() / 2, mean + offset,
                        f"{mean:+.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#333333")

        ax.set_title(f"True grade: {g}\n(n={n})", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([l for l, _, _ in CONDITIONS], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean grade inflation\n(awarded − baseline)", fontsize=10)
    fig.suptitle(f"{args.model} Grade Inflation by Stakes Level and True Grade", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if args.out_png:
        out_path = Path(args.out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
