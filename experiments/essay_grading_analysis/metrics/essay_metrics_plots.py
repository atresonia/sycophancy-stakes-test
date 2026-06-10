"""Forest-plot views of grade inflation (GI) for the essay-grading scenarios.

Reads the CSVs written by essay_metrics_scenarios.py and draws point-estimate +
95% CI bracket charts so the results can be shared without reading the tables.

GI = grade - kb_grade, where kb_grade is the model's own no-context baseline.
  GI < 0  -> social prompt makes the model grade HARSHER than baseline (anti-sycophantic)
  GI > 0  -> social prompt makes the model inflate (sycophantic)
A CI bracket that crosses 0 means the effect is not distinguishable from "no change".

Usage:
  python experiments/essay_grading_analysis/metrics/essay_metrics_plots.py \
      --metrics_dir data/essay_grading/metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

POS = "#2c7fb8"   # sycophantic (GI > 0)
NEG = "#d95f0e"   # anti-sycophantic (GI < 0)
CROSSES = "#9e9e9e"  # CI crosses 0 -> not distinguishable from no effect


def _color(row: pd.Series) -> str:
    if row["ci_lo"] <= 0 <= row["ci_hi"]:
        return CROSSES
    return POS if row["mean"] > 0 else NEG


def forest(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """One horizontal point + CI-bracket per row, sorted most-negative at bottom.

    df needs columns: mean, ci_lo, ci_hi, _label.
    """
    df = df.sort_values("mean").reset_index(drop=True)
    y = range(len(df))
    colors = [_color(r) for _, r in df.iterrows()]

    fig, ax = plt.subplots(figsize=(9, 0.46 * len(df) + 1.4))
    for yi, (_, r), c in zip(y, df.iterrows(), colors):
        ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=c, lw=2, zorder=2)
        # bracket caps
        for x in (r["ci_lo"], r["ci_hi"]):
            ax.plot([x, x], [yi - 0.16, yi + 0.16], color=c, lw=2, zorder=2)
        ax.scatter([r["mean"]], [yi], color=c, s=46, zorder=3)
        ax.annotate(f'{r["mean"]:+.2f}', (r["mean"], yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=c)

    ax.axvline(0, color="black", lw=1, ls="--", zorder=1)
    handles = [
        plt.Line2D([], [], color=NEG, marker="o", lw=2, label="anti-syc (CI below 0)"),
        plt.Line2D([], [], color=POS, marker="o", lw=2, label="sycophantic (CI above 0)"),
        plt.Line2D([], [], color=CROSSES, marker="o", lw=2, label="CI crosses 0 (no clear effect)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=False)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["_label"].tolist(), fontsize=9)
    ax.set_xlabel("Grade inflation (GI) vs model baseline   ← harsher / anti-syc      inflates / syc →")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.margins(x=0.12)
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ordered levels per leveled scenario; inner lists are connected as one trend line,
# so a scenario with two lists (feedback_actionability) draws two separate segments.
TREND_GROUPS = {
    "grade_weight": [["low", "medium", "high"]],
    "emotional_investment": [["low", "medium", "high"]],
    "time_spent": [["low", "medium", "high"]],
    "feedback_actionability": [["low", "medium", "high"], ["not_submitted", "submitted"]],
    "deserved_grade": [["A", "B", "C", "D", "F"]],
}
TREND_XLABEL = {
    "grade_weight": "stakes:  10% → 25% → 90% of grade",
    "emotional_investment": "investment:  low → high",
    "time_spent": "time spent:  month → week → hour",
    "feedback_actionability": "deadline: month→week→hour      |  submission",
    "deserved_grade": "grade the user claims they deserve",
}


def trend_facets(by_lvl: pd.DataFrame, out_path: Path) -> None:
    """Small-multiples: one panel per leveled scenario, GI across ordered levels."""
    lv = by_lvl[by_lvl["level"] != "(none)"].set_index(["scenario", "level"])["mean"]
    scen_order = list(TREND_GROUPS)
    ylo, yhi = lv.min() - 0.25, lv.max() + 0.25

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    axes = axes.ravel()
    for ax, scen in zip(axes, scen_order):
        pos, ticks, ticklabels = 0, [], []
        for group in TREND_GROUPS[scen]:
            xs, ys = [], []
            for lvl in group:
                if (scen, lvl) in lv.index:
                    y = lv.loc[(scen, lvl)]
                    xs.append(pos); ys.append(y)
                    ticks.append(pos); ticklabels.append(lvl.replace("_", "\n"))
                    color = NEG if y < 0 else POS
                    ax.annotate(f"{y:+.2f}", (pos, y), textcoords="offset points",
                                xytext=(0, 9), ha="center", fontsize=8, color=color)
                    ax.scatter([pos], [y], color=color, s=44, zorder=3)
                    pos += 1
            ax.plot(xs, ys, color="#9e9e9e", lw=1.8, zorder=2)
            pos += 1.4  # gap before next segment
        ax.axhline(0, color="black", lw=1, ls="--", zorder=1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=8.5)
        ax.set_title(scen, fontsize=11, fontweight="bold")
        ax.set_xlabel(TREND_XLABEL[scen], fontsize=8, color="#666")
        ax.set_ylim(ylo, yhi)
        ax.margins(x=0.12)
        ax.grid(axis="y", color="#eeeeee", zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[len(scen_order):]:
        ax.set_visible(False)

    fig.suptitle("Grade inflation trend within each scenario   (GI vs model baseline; below 0 = harsher)",
                 fontsize=12, fontweight="bold")
    fig.supylabel("mean grade inflation (GI)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", type=Path, default=Path("data/essay_grading/metrics"))
    p.add_argument("--out_dir", type=Path, default=None, help="defaults to <metrics_dir>/figs")
    args = p.parse_args()
    out_dir = args.out_dir or (args.metrics_dir / "figs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) by scenario
    by_scen = pd.read_csv(args.metrics_dir / "gi_by_scenario.csv")
    by_scen["_label"] = by_scen.apply(lambda r: f'{r["scenario"]}  (n={r["n"]})', axis=1)
    forest(by_scen,
           "Grade inflation by scenario  (point = mean, bracket = 95% CI)",
           out_dir / "gi_by_scenario.png")

    # 2) trend within each leveled scenario (small multiples, no CIs)
    by_lvl = pd.read_csv(args.metrics_dir / "gi_by_scenario_level.csv")
    trend_facets(by_lvl, out_dir / "gi_trend_by_level.png")


if __name__ == "__main__":
    main()
