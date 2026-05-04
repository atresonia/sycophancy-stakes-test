"""
Plot mean LLM grade across stakes levels with one line per model.

Usage:
    python scripts/utility_analysis/plot_multi_model_stakes.py \
        --in_json gemini:data/essay_grading/llm_resp_stakes_gemini_400_500.jsonl \
                  openai:data/essay_grading/llm_resp_stakes_openai_400_500.jsonl \
                  claude:data/essay_grading/llm_resp_stakes_claude_400_500.jsonl \
        --out_png plots/multi_model_stakes.png

Each --in_json entry is "label:path". Lines connect mean numeric grade
(A=5 … F=1) at each stakes level. Annotate any model run with the old
prompt using --note, e.g. --note "gemini:pre-calibration prompt".
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRADE_NUM = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
NUM_GRADE  = {v: k for k, v in GRADE_NUM.items()}

STAKES_GROUPS = {
    "baseline": ["baseline"],
    "low_stakes": ["low_stakes_1", "low_stakes_2", "low_stakes_3"],
    "high_stakes": ["high_stakes_1", "high_stakes_2", "high_stakes_3"],
}
STAKES_ORDER  = ["baseline", "low_stakes", "high_stakes"]
STAKES_LABELS = ["Baseline", "Low", "High"]

MODEL_COLORS = ["#4C9BE8", "#E84C4C", "#2CA02C", "#FF7F0E", "#9467BD"]


def _nearest_letter(val: float) -> str:
    base = max(1, min(5, round(val)))
    return NUM_GRADE[base]


def load_means(jsonl_path: Path) -> tuple[list[float], int]:
    """Return (means_per_stakes_level, n_essays)."""
    records: dict[str, list[float]] = {k: [] for k in STAKES_ORDER}
    essay_ids = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            essay_ids.add(rec.get("prompt_row_id"))
            for group, keys in STAKES_GROUPS.items():
                for key in keys:
                    output = rec.get(f"{key}_output")
                    if output and output.get("grade") in GRADE_NUM:
                        records[group].append(GRADE_NUM[output["grade"]])
    means = [float(np.mean(records[k])) if records[k] else float("nan") for k in STAKES_ORDER]
    return means, len(essay_ids)


def plot(model_data: dict[str, tuple[list[float], int]], notes: dict[str, str], out_path: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(STAKES_LABELS))

    for (label, (means, n)), color in zip(model_data.items(), MODEL_COLORS):
        note = notes.get(label, "")
        display = f"{label} ({note})" if note else label
        ax.plot(x, means, marker="o", linewidth=2, markersize=7, color=color, label=display)
        for xi, m in zip(x, means):
            if not np.isnan(m):
                ax.annotate(
                    _nearest_letter(m),
                    (xi, m),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color=color, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(STAKES_LABELS, fontsize=11)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["F", "D", "C", "B", "A"], fontsize=11)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("Stakes Level", fontsize=12)
    ax.set_ylabel("Mean Grade", fontsize=12)
    ax.set_title("Mean LLM Grade by Stakes Level", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    n_vals = [n for _, (_, n) in model_data.items()]
    if len(set(n_vals)) == 1:
        fig.text(0.5, -0.02, f"n = {n_vals[0]} essays per model", ha="center", fontsize=9, color="#666666")
    else:
        ns = ", ".join(f"{lbl}: {n}" for (lbl, (_, n)) in model_data.items())
        fig.text(0.5, -0.02, f"n = {ns}", ha="center", fontsize=9, color="#666666")

    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_json", nargs="+", required=True,
        metavar="LABEL:PATH",
        help="One or more label:path pairs, e.g. gemini:data/.../llm_resp_stakes_gemini.jsonl",
    )
    parser.add_argument("--out_png", default=None)
    parser.add_argument(
        "--note", nargs="*", default=[],
        metavar="LABEL:NOTE",
        help="Optional annotation per model, e.g. 'gemini:pre-calibration prompt'",
    )
    args = parser.parse_args()

    notes = {}
    for entry in args.note:
        label, _, note = entry.partition(":")
        notes[label.strip()] = note.strip()

    model_data = {}
    for entry in args.in_json:
        label, _, path_str = entry.partition(":")
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found for model '{label}': {path}")
        means, n = load_means(path)
        model_data[label.strip()] = (means, n)
        print(f"{label} ({n} essays):")
        for stakes_label, m in zip(STAKES_LABELS, means):
            if np.isnan(m):
                print(f"  {stakes_label:<10} N/A")
            else:
                print(f"  {stakes_label:<10} {_nearest_letter(m)} ({m:.3f})")

    plot(model_data, notes, Path(args.out_png) if args.out_png else None)


if __name__ == "__main__":
    main()
