"""Shareable tables of grade inflation (GI) for the essay-grading scenarios.

Produces two clean, self-explanatory tables (Markdown + PNG) from the CSVs that
essay_metrics_scenarios.py writes:

  Table 1  GI by scenario   -> scenario, mean [95% CI]
  Table 2  one row per scenario, with the level progression, the prompt that was
           added (the part that changes across levels collapsed to {a/b/c}), and
           the per-level GI.

The prompt text is read straight from essay_grading_scenarios.py (via AST, so the
module's heavy imports don't have to load), so it always matches what was actually run.

Usage:
  python experiments/essay_grading_analysis/metrics/essay_metrics_tables.py \
      --metrics_dir data/essay_grading/metrics
"""
from __future__ import annotations

import argparse
import ast
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCENARIOS_PY = Path("experiments/essay_grading_analysis/essay_grading_scenarios.py")

POS = "#2c7fb8"   # GI > 0 (sycophantic)
NEG = "#d95f0e"   # GI < 0 (anti-sycophantic)
CROSSES = "#9e9e9e"  # CI crosses 0 -> no clear effect
BLACK = "#222222"

LEVEL_RANK = {"low": 0, "medium": 1, "high": 2, "not_submitted": 3, "submitted": 4,
              "A": 0, "B": 1, "C": 2, "D": 3, "F": 4, "(none)": -1}

LEGEND = (
    "GI = the model's grade minus its own no-context baseline grade (A=4 … F=0). "
    "Negative = the message made it grade *harder*; positive = it inflated. "
    "CI = 95% confidence interval; if the CI crosses 0 the effect is not statistically clear. "
    "In Table 2, {a/b/c} marks the part of the prompt that changes across levels (same order as the Level column)."
)


def load_scenario_prompts(path: Path) -> dict[tuple[str, str], str]:
    """Return {(scenario, level): added_prompt_text} read from the scenarios source.

    Uses AST to evaluate only the prompt-template assignments and the SCENARIOS dict,
    so importing the module (and its boto3/genai/etc. deps) is unnecessary.
    """
    tree = ast.parse(path.read_text())
    wanted = {
        "VALID_GRADES", "STAKES_VARIANT_PROMPT_TEMPLATE",
        "FEEDBACK_ACTIONABILITY_PROMPT_TEMPLATE", "TIME_SPENT_PROMPT_TEMPLATE",
        "DESERVED_GRADE_PROMPT_TEMPLATE", "SCENARIOS",
    }
    body = [n for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)]
    ns: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)

    out: dict[tuple[str, str], str] = {}
    for name, scen in ns["SCENARIOS"].items():
        if "prompt" in scen:
            out[(name, "(none)")] = scen["prompt"]
        else:
            for level, variant in scen.items():
                out[(name, level)] = variant["prompt"]
    return out


def _mean_color(row: pd.Series) -> str:
    if row["ci_lo"] <= 0 <= row["ci_hi"]:
        return CROSSES
    return POS if row["mean"] > 0 else NEG


def _gi_ci(row: pd.Series) -> str:
    """Mean with the 95% CI in brackets, e.g. '-0.81 [-0.96, -0.67]'."""
    return f'{row["mean"]:+.2f}  [{row["ci_lo"]:+.2f}, {row["ci_hi"]:+.2f}]'


def _common_suffix(strs: list[str]) -> str:
    return os.path.commonprefix([s[::-1] for s in strs])[::-1]


def collapse_prompt(pairs: list[tuple[str, str]]) -> list[str]:
    """Collapse level prompts into one template, e.g. 'I spent {a month/a week/an hour} ...'.

    pairs: (level_label, prompt) in display order. Falls back to a 'level: prompt' list
    when the variants don't share a tight common template (returns one string per line).
    """
    prompts = [p for _, p in pairs]
    if len(prompts) == 1:
        return [prompts[0]]
    pre = os.path.commonprefix(prompts)
    pre = pre[:pre.rfind(" ") + 1] if " " in pre else ""   # word-align back to last space
    suf = _common_suffix(prompts)
    mids = [p[len(pre): len(p) - len(suf)] for p in prompts]
    tight = (len(pre) + len(suf) >= 8 and all(mids)
             and max(len(m) for m in mids) <= 20 and len(set(mids)) == len(mids))
    if tight:
        return [f"{pre}{{{'/'.join(mids)}}}{suf}"]
    return [f"{lvl}: {p}" for lvl, p in pairs]


def build_scenario_rows(t2: pd.DataFrame) -> list[dict]:
    """One dict per scenario: level_str, prompt_lines, gi_lines [(text, color)]."""
    rows = []
    for scen, grp in t2.groupby("scenario", sort=False):
        pairs = [(r["level"], r["prompt"]) for _, r in grp.iterrows()]
        single = len(grp) == 1 and grp.iloc[0]["level"] == "(none)"
        level_str = "—" if single else " → ".join(r["level"] for _, r in grp.iterrows())
        prompt_lines = collapse_prompt(pairs)
        if single:
            r = grp.iloc[0]
            gi_lines = [(_gi_ci(r), _mean_color(r))]
        else:
            gi_lines = [(f'{r["level"]}:  {_gi_ci(r)}', _mean_color(r)) for _, r in grp.iterrows()]
        rows.append({"scenario": scen, "level_str": level_str,
                     "prompt_lines": prompt_lines, "gi_lines": gi_lines})
    return rows


# ----------------------------------------------------------------------------- markdown

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = lambda cells: "| " + " | ".join(cells) + " |"
    out = [line(headers), line(["---"] * len(headers))]
    out += [line(r) for r in rows]
    return "\n".join(out)


def write_markdown(t1: pd.DataFrame, scen_rows: list[dict], out_path: Path) -> None:
    rows1 = [[r["scenario"], _gi_ci(r)] for _, r in t1.iterrows()]
    rows2 = [[s["scenario"], s["level_str"], "<br>".join(s["prompt_lines"]),
              "<br>".join(t for t, _ in s["gi_lines"])] for s in scen_rows]
    md = [
        "# Grade inflation by scenario", "",
        f"> {LEGEND}", "",
        "## Table 1 — Grade inflation by scenario", "",
        _md_table(["Scenario", "Mean GI [95% CI]"], rows1), "",
        "## Table 2 — What each scenario adds, by level", "",
        _md_table(["Scenario", "Level", "Added to the prompt", "Mean GI [95% CI]"], rows2), "",
    ]
    out_path.write_text("\n".join(md))
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------------- png

def _cell_lines(val) -> list[tuple[str, str]]:
    """Normalize a cell to a list of (text, color) lines."""
    if isinstance(val, list):
        return list(val)
    if isinstance(val, tuple):
        return [val]
    return [(str(val), BLACK)]


def render_table_png(headers, rows, fracs, aligns, wrap_chars, out_path, title, fig_w=11.5):
    """Draw a shaded table as a PNG. A cell may be a str, (text, color), or a list of
    (text, color) lines; lines align row-by-row across columns."""
    wrapped, rowlines = [], []
    for row in rows:
        cells, nlines = [], 1
        for j, val in enumerate(row):
            lines = []
            for text, color in _cell_lines(val):
                for seg in str(text).split("\n"):
                    pieces = textwrap.wrap(seg, wrap_chars[j]) if wrap_chars[j] else [seg]
                    lines += [(p, color) for p in (pieces or [""])]
            cells.append(lines)
            nlines = max(nlines, len(lines))
        wrapped.append(cells)
        rowlines.append(nlines)

    title_u, header_u, line_u = 1.6, 1.4, 0.95
    row_u = [n * line_u + 0.55 for n in rowlines]
    total_u = title_u + header_u + sum(row_u)
    fig, ax = plt.subplots(figsize=(fig_w, total_u * 0.27))
    ax.set_xlim(0, 1); ax.set_ylim(0, total_u); ax.axis("off")
    ax.invert_yaxis()  # draw top-down

    x_left = [sum(fracs[:j]) for j in range(len(fracs))]
    pad = 0.008

    def cell_x(j, align):
        if align == "right":
            return x_left[j] + fracs[j] - pad, "right"
        if align == "center":
            return x_left[j] + fracs[j] / 2, "center"
        return x_left[j] + pad, "left"

    y = 0.0
    ax.text(0.0, y + title_u * 0.55, title, fontsize=13, fontweight="bold", va="center")
    y += title_u

    ax.add_patch(plt.Rectangle((0, y), 1, header_u, color="#2b2b2b"))
    for j, h in enumerate(headers):
        hx, ha = cell_x(j, aligns[j])
        ax.text(hx, y + header_u / 2, h, color="white", fontsize=10.5,
                fontweight="bold", va="center", ha=ha)
    y += header_u

    for i, (cells, ru) in enumerate(zip(wrapped, row_u)):
        if i % 2:
            ax.add_patch(plt.Rectangle((0, y), 1, ru, color="#f5f5f5"))
        for j, lines in enumerate(cells):
            cx, ha = cell_x(j, aligns[j])
            for k, (text, color) in enumerate(lines):
                ax.text(cx, y + 0.45 + k * line_u, text, color=color,
                        fontsize=9.5, va="top", ha=ha)
        ax.plot([0, 1], [y, y], color="#dddddd", lw=0.6)
        y += ru

    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", type=Path, default=Path("data/essay_grading/metrics"))
    p.add_argument("--scenarios_py", type=Path, default=SCENARIOS_PY)
    p.add_argument("--out_dir", type=Path, default=None, help="defaults to <metrics_dir>/tables")
    args = p.parse_args()
    out_dir = args.out_dir or (args.metrics_dir / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_scenario_prompts(args.scenarios_py)

    # Table 1: by scenario, sorted most-negative first.
    t1 = (pd.read_csv(args.metrics_dir / "gi_by_scenario.csv")[["scenario", "mean", "ci_lo", "ci_hi"]]
          .sort_values("mean").reset_index(drop=True))

    # Table 2: by scenario x level, ordered by scenario (table-1 order) then level.
    t2 = pd.read_csv(args.metrics_dir / "gi_by_scenario_level.csv")
    t2["prompt"] = t2.apply(lambda r: prompts.get((r["scenario"], r["level"]), ""), axis=1)
    scen_rank = {s: i for i, s in enumerate(t1["scenario"])}
    t2["_s"] = t2["scenario"].map(scen_rank)
    t2["_l"] = t2["level"].map(lambda x: LEVEL_RANK.get(x, 99))
    t2 = t2.sort_values(["_s", "_l"]).reset_index(drop=True)

    # display: underscores -> spaces (cleaner wrapping; avoids stray Markdown italics)
    t1["scenario"] = t1["scenario"].str.replace("_", " ")
    t2["scenario"] = t2["scenario"].str.replace("_", " ")
    t2["level"] = t2["level"].str.replace("_", " ")

    scen_rows = build_scenario_rows(t2)

    write_markdown(t1, scen_rows, out_dir / "essay_gi_tables.md")

    # Table 1 PNG
    rows1 = [[r["scenario"], (_gi_ci(r), _mean_color(r))] for _, r in t1.iterrows()]
    render_table_png(
        ["Scenario", "Mean GI  [95% CI]"], rows1,
        fracs=[0.5, 0.5], aligns=["left", "left"], wrap_chars=[0, 0],
        out_path=out_dir / "gi_by_scenario.png",
        title="Grade inflation by scenario  (GI vs model baseline)")

    # Table 2 PNG (one row per scenario)
    rows2 = [[s["scenario"], s["level_str"], "\n".join(s["prompt_lines"]), s["gi_lines"]]
             for s in scen_rows]
    render_table_png(
        ["Scenario", "Level", "Added to the prompt", "Mean GI  [95% CI]"], rows2,
        fracs=[0.15, 0.15, 0.42, 0.28],
        aligns=["left", "left", "left", "left"], wrap_chars=[14, 13, 46, 0],
        out_path=out_dir / "gi_by_scenario_level.png",
        title="What each scenario adds to the prompt, and its grade inflation")

    print(f"\n{LEGEND}")


if __name__ == "__main__":
    main()
