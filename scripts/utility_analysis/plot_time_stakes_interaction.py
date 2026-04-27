"""
Interaction plot for the 2x2 Time x Stakes experiment.

Usage:
    python scripts/utility_analysis/plot_time_stakes_interaction.py \
        --in_json gemini:data/essay_grading/llm_resp_time_stakes_gemini.jsonl \
                  openai:data/essay_grading/llm_resp_time_stakes_openai.jsonl \
                  claude:data/essay_grading/llm_resp_time_stakes_claude.jsonl \
        --out_png plots/time_stakes_interaction.png

--in_json takes 1+ label:path pairs.

Grade inflation per essay per cell = cell_grade_num - baseline_grade_num.
Plot shows mean inflation for each of the four cells:
  (low, short), (low, long), (high, short), (high, long)
stdout format:
gemini (n=100)
                       Short      Long         Δ
Low stakes             -0.11     -0.11     +0.00
High stakes            -0.07     -0.18     -0.11
Δ(High-Low)            +0.04     -0.07
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Resolve the utils package relative to this script's location
sys.path.insert(0, str(Path(__file__).parent))
from utils.utility import GRADE_NUM, load_jsonl

LOW_COLOR = "#5b9bd5"
HIGH_COLOR = "#c0392b"

CELLS = [
    ("low_short", "low_short_output"),
    ("low_long",  "low_long_output"),
    ("high_short", "high_short_output"),
    ("high_long",  "high_long_output"),
]


def compute_inflations(records: list[dict]) -> dict[str, list[float]]:
    """Return dict of cell -> list of per-essay inflation values."""
    inflations: dict[str, list[float]] = {name: [] for name, _ in CELLS}
    for rec in records:
        if rec.get("status") != "ok":
            continue
        baseline_out = rec.get("baseline_output")
        if not baseline_out or baseline_out.get("grade") not in GRADE_NUM:
            continue
        baseline_num = GRADE_NUM[baseline_out["grade"]]
        for cell_name, field in CELLS:
            out = rec.get(field)
            if out and out.get("grade") in GRADE_NUM:
                inflations[cell_name].append(GRADE_NUM[out["grade"]] - baseline_num)
    return inflations


def print_summary(label: str, inflations: dict[str, list[float]]) -> None:
    n = min(len(inflations[name]) for name, _ in CELLS)
    low_short  = np.mean(inflations["low_short"])
    low_long   = np.mean(inflations["low_long"])
    high_short = np.mean(inflations["high_short"])
    high_long  = np.mean(inflations["high_long"])

    delta_low  = low_long  - low_short
    delta_high = high_long - high_short
    delta_short = high_short - low_short
    delta_long  = high_long  - low_long

    print(f"{label} (n={n})")
    print(f"{'':18s}{'Short':>10}{'Long':>10}{'Δ':>10}")
    print(f"{'Low stakes':<18s}{low_short:>+10.2f}{low_long:>+10.2f}{delta_low:>+10.2f}")
    print(f"{'High stakes':<18s}{high_short:>+10.2f}{high_long:>+10.2f}{delta_high:>+10.2f}")
    print(f"{'Δ(High-Low)':<18s}{delta_short:>+10.2f}{delta_long:>+10.2f}")
    print()


def plot(model_data: dict[str, dict[str, list[float]]], out_path: Path | None) -> None:
    n_models = len(model_data)
    fig_width = max(5 * n_models, 5)
    fig, axes = plt.subplots(1, n_models, figsize=(fig_width, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    x = np.array([0, 1])
    x_labels = ["Short (1 hr)", "Long (1 mo)"]

    all_means = []

    # First pass: collect all means to determine y range
    panels = []
    for label, inflations in model_data.items():
        low_means  = [np.mean(inflations["low_short"]),  np.mean(inflations["low_long"])]
        high_means = [np.mean(inflations["high_short"]), np.mean(inflations["high_long"])]
        n = min(len(inflations[name]) for name, _ in CELLS)
        panels.append((label, low_means, high_means, n))
        all_means.extend(low_means + high_means)

    # Guard against empty data (all NaN means)
    finite_means = [v for v in all_means if np.isfinite(v)]
    if not finite_means:
        print("ERROR: No valid data found. Check that your JSONL files contain "
              "time×stakes fields (low_short_output, low_long_output, "
              "high_short_output, high_long_output).\n"
              "Regular stakes files (low_stakes_1_output, etc.) are not compatible "
              "with this script.")
        sys.exit(1)

    pad = 0.3
    y_min = min(finite_means) - pad
    y_max = max(finite_means) + pad

    # Integer ticks spanning the data range
    tick_min = int(np.floor(min(finite_means)))
    tick_max = int(np.ceil(max(finite_means)))
    yticks = list(range(tick_min, tick_max + 1))

    for ax, (label, low_means, high_means, n) in zip(axes, panels):
        # Low stakes line
        ax.plot(x, low_means, color=LOW_COLOR, linewidth=2,
                marker="o", markersize=8, label="Low stakes")
        # High stakes line
        ax.plot(x, high_means, color=HIGH_COLOR, linewidth=2,
                marker="s", markersize=8, label="High stakes")

        # Annotate each point
        for xi, val in zip(x, low_means):
            offset = 12 if val >= high_means[xi] else -16
            ax.annotate(f"{val:+.2f}", (xi, val),
                        textcoords="offset points", xytext=(0, offset),
                        ha="center", fontsize=9, color=LOW_COLOR)
        for xi, val in zip(x, high_means):
            offset = 12 if val >= low_means[xi] else -16
            ax.annotate(f"{val:+.2f}", (xi, val),
                        textcoords="offset points", xytext=(0, offset),
                        ha="center", fontsize=9, color=HIGH_COLOR)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_yticks(yticks)
        ax.set_ylim(y_min, y_max)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Grade inflation vs. baseline", fontsize=10)

    # Legend on first panel only
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)

    # Count: use smallest n across models for footer
    n_vals = [n for _, _, _, n in panels]
    if len(set(n_vals)) == 1:
        footer = f"n = {n_vals[0]} essays"
    else:
        footer = "n = " + ", ".join(
            f"{lbl}: {n}" for (lbl, _, _, n) in panels
        )
    fig.text(0.5, -0.02, footer, ha="center", fontsize=9, color="#666666")

    fig.suptitle("Grade Inflation: Time \u00d7 Stakes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved \u2192 {out_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_json", nargs="+", required=True,
        metavar="LABEL:PATH",
        help="One or more label:path pairs",
    )
    parser.add_argument("--out_png", default=None)
    args = parser.parse_args()

    model_data: dict[str, dict[str, list[float]]] = {}
    for entry in args.in_json:
        label, _, path_str = entry.partition(":")
        label = label.strip()
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found for model '{label}': {path}")
        records = load_jsonl(path)
        inflations = compute_inflations(records)
        model_data[label] = inflations
        print_summary(label, inflations)

    plot(model_data, Path(args.out_png) if args.out_png else None)


if __name__ == "__main__":
    main()
