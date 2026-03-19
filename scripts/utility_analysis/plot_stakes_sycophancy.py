"""
Plot mean LLM grade across stakes levels from llm_resp_stakes_*.jsonl.

Produces a single bar chart: mean grade (A–F) for Baseline / Low / Medium / High stakes.

Usage:
    python scripts/utility_analysis/plot_stakes_sycophancy.py \
        --in_json data/essay_grading/llm_resp_stakes_gemini.jsonl \
        --out_png  plots/stakes_sycophancy_gemini.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRADE_NUM = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
NUM_TO_GRADE = {5: "A", 4: "B", 3: "C", 2: "D", 1: "F"}

# Breakpoints for +/− modifiers. Each grade spans 1 unit (e.g. D = 2.0).
# Lower third → base−, middle third → base, upper third → base+
# Exception: A has no A+; F has no F−.
def num_to_letter_mod(val: float) -> str:
    base = round(val)
    base = max(1, min(5, base))
    letter = NUM_TO_GRADE[base]
    frac = val - base          # how far above/below the nearest integer
    if frac >= 0.34 and letter != "A":
        modifier = "+"
    elif frac <= -0.34 and letter != "F":
        modifier = "−"
    else:
        modifier = ""
    return f"{letter}{modifier}"
STAKES_KEYS = ["baseline", "low_stakes", "medium_stakes", "high_stakes"]
STAKES_LABELS = ["Baseline", "Low", "Medium", "High"]
STAKES_COLORS = ["#888888", "#4C9BE8", "#F5A623", "#E84C4C"]


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
                    "given_num": GRADE_NUM[output["grade"]],
                    "gt_num": GRADE_NUM[gt],
                })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    means = []
    for label in STAKES_LABELS:
        vals = df.loc[df["stakes"] == label, "given_num"].values
        means.append(float(np.mean(vals)))

    bars = ax.bar(
        STAKES_LABELS, means,
        color=STAKES_COLORS, width=0.5,
        alpha=0.88, edgecolor="white", linewidth=1.2, zorder=3,
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, m + 0.06,
            num_to_letter_mod(m), ha="center", va="bottom", fontsize=12, fontweight="bold", color="#222222",
        )

    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["F", "D", "C", "B", "A"], fontsize=11)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("Stakes Level", fontsize=12)
    ax.set_ylabel("Mean LLM Grade", fontsize=12)
    ax.set_title("Mean LLM Grade by Stakes Level", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    n_essays = df["prompt_row_id"].nunique()
    fig.text(0.5, -0.03, f"n = {n_essays} essays", ha="center", fontsize=9, color="#666666")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved plot → {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot mean grade by stakes level")
    parser.add_argument("--in_json", default="data/essay_grading/llm_resp_stakes_gemini.jsonl")
    parser.add_argument("--out_png", default="plots/stakes_sycophancy_gemini.png")
    parser.add_argument("--use_llm_as_ground_truth", "-llm", action="store_true", default=False)
    args = parser.parse_args()

    in_path = Path(args.in_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = load_records(in_path, use_llm_as_ground_truth=args.use_llm_as_ground_truth)
    print(f"Loaded {df['prompt_row_id'].nunique()} essays, {len(df)} stake-condition records\n")

    print(f"{'Stakes':<10}  {'Grade':>6}  {'n':>5}")
    print("-" * 26)
    for label in STAKES_LABELS:
        sub = df[df["stakes"] == label]
        if sub.empty:
            continue
        m = sub["given_num"].mean()
        print(f"{label:<10}  {num_to_letter_mod(m):>6}  {len(sub):>5}")

    print()
    plot(df, Path(args.out_png))


if __name__ == "__main__":
    main()
