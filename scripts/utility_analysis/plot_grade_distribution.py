"""
Stacked bar chart: given-grade distribution conditioned on ground-truth grade.

One subplot per ground-truth grade (A/B/C/D/F).  Each subplot has one stacked bar
per stakes level (Baseline/Low/Medium/High) showing the % of essays that received
each given grade (A/B/C/D/F).

Ground truth source is shown in the plot title:
  - Default:  raw ASAP grader score (converted to letter)
  - --use_llm_as_ground_truth: LLM's own baseline grade

Usage:
    python scripts/utility_analysis/plot_grade_distribution.py \
        --in_json data/essay_grading/llm_resp_stakes_gemini.jsonl \
        --out_png  plots/grade_distribution_gemini.png

    # Use LLM grade as ground truth instead of human grader
    python scripts/utility_analysis/plot_grade_distribution.py \
        --in_json data/essay_grading/llm_resp_stakes_gemini.jsonl \
        --out_png  plots/grade_distribution_gemini_llm_gt.png \
        --use_llm_as_ground_truth
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


GRADE_NUM = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
GRADE_COLORS = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}
GRADES_ORDERED = ["A", "B", "C", "D", "F"]
STAKES_KEYS = ["baseline", "low_stakes", "medium_stakes", "high_stakes"]
STAKES_LABELS = ["Baseline", "Low", "Medium", "High"]


def load_records(jsonl_path: Path, use_llm_as_ground_truth: bool = False) -> pd.DataFrame:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            gt_field = "llm_response" if use_llm_as_ground_truth else "ground_truth"
            gt = rec.get(gt_field, "")
            if gt not in GRADE_NUM:
                continue
            for key, label in zip(STAKES_KEYS, STAKES_LABELS):
                output = rec.get(f"{key}_output")
                if output is None or output.get("grade") not in GRADE_NUM:
                    continue
                rows.append({
                    "prompt_row_id": rec.get("prompt_row_id"),
                    "ground_truth": gt,
                    "stakes": label,
                    "given_grade": output["grade"],
                })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path, gt_source_label: str) -> None:
    gt_grades = [g for g in GRADES_ORDERED if g in df["ground_truth"].unique()]
    n_cols = len(gt_grades)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        f"Given Grade Distribution by Stakes Level\nGround truth: {gt_source_label}",
        fontsize=13, fontweight="bold", y=1.04,
    )

    x = range(len(STAKES_LABELS))

    for ax, gt in zip(axes, gt_grades):
        df_gt = df[df["ground_truth"] == gt]
        n = df_gt["prompt_row_id"].nunique()
        ax.set_title(f"True grade = {gt}\n(n={n})", fontsize=11, fontweight="semibold")

        bottoms = [0.0] * len(STAKES_LABELS)
        for given_grade in GRADES_ORDERED:
            heights = []
            for label in STAKES_LABELS:
                sub = df_gt[df_gt["stakes"] == label]
                pct = (sub["given_grade"] == given_grade).mean() * 100 if len(sub) > 0 else 0.0
                heights.append(pct)

            bars = ax.bar(
                STAKES_LABELS, heights,
                bottom=bottoms,
                color=GRADE_COLORS[given_grade],
                width=0.55,
                edgecolor="white",
                linewidth=0.8,
            )
            # Label segments that are large enough to read
            for bar, h, b in zip(bars, heights, bottoms):
                if h >= 8:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        b + h / 2,
                        f"{given_grade}\n{h:.0f}%",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white",
                    )
            bottoms = [b + h for b, h in zip(bottoms, heights)]

        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(STAKES_LABELS)))
        ax.set_xticklabels(STAKES_LABELS, fontsize=9, rotation=15, ha="right")
        ax.set_xlabel("Stakes Level", fontsize=10)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("% of Essays", fontsize=10)

    legend_patches = [mpatches.Patch(color=GRADE_COLORS[g], label=f"Given {g}") for g in GRADES_ORDERED]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.1), framealpha=0.9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved plot → {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Stacked bar: grade distribution across stakes levels")
    parser.add_argument("--in_json", default="data/essay_grading/llm_resp_stakes_gemini.jsonl")
    parser.add_argument("--out_png", default="plots/grade_distribution_gemini.png")
    parser.add_argument(
        "--use_llm_as_ground_truth", "-llm", action="store_true", default=False,
        help="Use the LLM's own grade as ground truth instead of the human grader score",
    )
    args = parser.parse_args()

    in_path = Path(args.in_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    gt_source_label = "LLM baseline grade" if args.use_llm_as_ground_truth else "human grader (ASAP score)"

    df = load_records(in_path, use_llm_as_ground_truth=args.use_llm_as_ground_truth)
    print(f"Loaded {df['prompt_row_id'].nunique()} essays, {len(df)} stake-condition records")
    print(f"Ground truth source: {gt_source_label}\n")

    plot(df, Path(args.out_png), gt_source_label)


if __name__ == "__main__":
    main()
