"""
Summarize results from essay grading, MMLU, and Elephant experiments.

Produces for each experiment:
 - per-condition table: primary-metric mean, 95% CI, n
 - precondition test: baseline sycophancy magnitude, p, Cohen's d
 - experiment-specific contribution metric
 - a condition-means plot (one PNG per experiment)
Usage:
  python summarize.py --experiment essay   --results_csv R.csv --ground_truth_csv GT.csv [--plot_out fig.png]
  python summarize.py --experiment mmlu    --results_csv R.csv [--plot_out fig.png]
  python summarize.py --experiment elephant --results_csv R.csv [--plot_out fig.png]
"""
import argparse
import os
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
 
import experiments.essay_grading_analysis.metrics.essay_metrics as essay_metrics
import experiments.mmlu_analysis.metrics.mmlu_metrics as mmlu_metrics
import experiments.elephant_analysis.metrics.elephant_metrics as elephant_metrics


def load_fit_result(experiment, model, fit_results_csv):
    """Look up B_stakes row for (experiment, model) in fit_results.csv."""
    if fit_results_csv is None or not os.path.exists(fit_results_csv):
        print(f"WARNING: no fit_results.csv found for {experiment}/{model}")
        return None
    
    fr_df = pd.read_csv(fit_results_csv)
    fit = fr_df[(fr_df["experiment"] == experiment) & (fr_df["model"] == model)]
    if len(fit) == 0:
        print(f"WARNING: no fit_results.csv row found for {experiment}/{model}")
        return None
    r = fit.iloc[-1]
    return {
        "beta": r["beta"],
        "se": r["se"],
        "p_one_sided": r["p_one_sided"],
        "ci_lo": r["ci_lo"],
        "ci_hi": r["ci_hi"],
        "degenerate": bool(r["degenerate"]),
    }


def print_bstakes(experiment, model, fit_results_csv, scale_note=""):
    """Print B_stakes from fit_results.csv. Placeholder if absent."""
    fit = load_fit_result(experiment, model, fit_results_csv)
    if fit is None:
        print(f"\nB_stakes [{experiment}/{model}] -- no fit_results.csv row. "
              f"Run the fitting script, or paste by hand:")
        print("  B_stakes [ ____ ]  p_one-sided [ ____ ]  95% CI [ ____ ]")
        return
    deg = ("  *** DEGENERATE FIT -- unreliable, use the glm fallback ***"
           if fit["degenerate"] else "")
    print(f"\nB_stakes [{experiment}/{model}]{scale_note}:  "
          f"{fit['beta']:+.4f}  SE {fit['se']:.4f}  "
          f"p_one-sided {fit['p_one_sided']:.5e}  "
          f"95% CI [{fit['ci_lo']:+.4f}, {fit['ci_hi']:+.4f}]{deg}")


def print_table(title, header, rows):
    print(f"\n{title}")
    widths = [max(len(str(header[i])), *(len(str(r[i])) for r in rows))
              for i in range(len(header))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


def plot(labels, means, ci_lo, ci_hi, ylabel, title, zero_line, out_png, n_list=None):
    """Condition-means plot: point estimate +/- 95% CI per condition."""
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = range(len(labels))
    yerr = [[m - lo for m, lo in zip(means, ci_lo)],
            [hi - m for m, hi in zip(means, ci_hi)]]
    ax.errorbar(x, means, yerr=yerr, fmt="o", capsize=3,
                color="black", ecolor="gray")
    if zero_line is not None:
        ax.axhline(zero_line, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    if n_list:
        lo_n, hi_n = min(n_list), max(n_list)
        title = f"{title} (n = {lo_n})"
        # threshold: warn if high range
        spread = hi_n - lo_n
        if hi_n and (spread / hi_n > 0.05 or spread > 20):
            print(f"  WARNING: per-condition n varies {lo_n}-{hi_n} "
                  f"(spread {spread}). The plot title shows the minimum; "
                  f"this spread may indicate differential attrition -- "
                  f"check the per-condition n column.")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", dpi=300)
        print(f"\nWrote plot -> {out_png}")
    else:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# ESSAY
# -----------------------------------------------------------------------------
def summarize_essay(results_csv, ground_truth_csv, plot_out, model, fit_results_csv):
    df = essay_metrics.load_results(results_csv, ground_truth_csv)
    gi = essay_metrics.calculate_grade_inflation(df)
    pr = essay_metrics.progressive_vs_regressive(df)
    pc = essay_metrics.precondition_baseline_sycophancy(df)
 
    order = [("user_sycophancy", "User syc."),
             ("low_stakes", "Low"),
             ("medium_stakes", "Medium"),
             ("high_stakes", "High")]
    n_list = [gi[k]["n"] for k, _ in order]
    print("=" * 70)
    print("ESSAY GRADING STAKES -- PAPER SUMMARY")
    print("=" * 70)
    print("Primary metric: grade shift GI = grade(cond) - grade(kb). "
          "+ve = inflation.")

    # mean knowledge-baseline grade (GI is measured against this so the KB's GI is 0)
    kb_mean = df["kb_num"].mean()
    gt_mean = df["gt_num"].mean()
    # print the number and letter grade estimate for the KB and GT means
    print(f"Knowledge-baseline grade: {kb_mean:.3f} (estimate: {essay_metrics.to_letter(kb_mean)})")
    print(f"Ground-truth grade: {gt_mean:.3f} (estimate: {essay_metrics.to_letter(gt_mean)})")
 
    # per-condition table
    rows = []
    for key, disp in order:
        s = gi[key]
        lo, hi = s["ci_95"]
        p = pr[key]
        rows.append([disp, f"{s['mean']:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", str(s["n"]),
                     f"{p['ProgRate']:.3f}", f"{p['RegRate']:.3f}"])
    print_table("PER-CONDITION (GI mean, 95% CI, n, Prog/Reg rate):",
                 ["Condition", "GI mean", "95% CI", "n", "ProgRate", "RegRate"],
                 rows)
 
    # precondition
    print("\nPRECONDITION (baseline sycophancy, GI(user) > 0):")
    print(f"  GI(user) = {pc['gi_mean']:+.3f}  d = {pc['cohens_d']:.3f}  "
          f"p_one-sided = {pc['p_one_sided']:.5e}  "
          f"(report as a magnitude, not a pass/fail)")
 
    # cross-experiment comparable number: stakes 0->2 change in the raw metric
    delta = gi["high_stakes"]["mean"] - gi["low_stakes"]["mean"]
    print(f"\nSTAKES 0->2 IMPLIED CHANGE (high - low, grade units): {delta:+.3f}")
    print("  -- the cross-experiment comparable number for essay")
 
    # B_stakes slot
    if fit_results_csv:
        print_bstakes("essay", model, fit_results_csv)
 
    # plot
    labels = [d for _, d in order]
    means = [gi[k]["mean"] for k, _ in order]
    lo = [gi[k]["ci_95"][0] for k, _ in order]
    hi = [gi[k]["ci_95"][1] for k, _ in order]
    plot(labels, means, lo, hi, "Grade shift (GI)",
          f"Essay: grade shift by condition ({model})", 0.0, plot_out, n_list=n_list)
    print("=" * 70)


# -----------------------------------------------------------------------------
# MMLU
# -----------------------------------------------------------------------------
def summarize_mmlu(results_csv, plot_out, model, fit_results_csv):
    import pandas as pd
    df = pd.read_csv(results_csv)
    m = mmlu_metrics.calculate_metrics(df)
    pc = mmlu_metrics.precondition_baseline_sycophancy(df)
 
    # primary model is fit on the three stakes levels, incorrect placement
    order = [("low_stakes", "Low"), ("medium_stakes", "Medium"),
             ("high_stakes", "High")]
 
    print("=" * 70)
    print("MMLU BET-STAKES -- PAPER SUMMARY")
    print("=" * 70)
    print("Primary metric: flip_to_user (model adopts the user's incorrect "
          "answer). Stakes levels only; baselines descriptive.")

    kb = m.get("knowledge_baseline")
    if kb is not None:
        print(f"Knowledge baseline (no user framing): "
              f"accuracy = {kb['accuracy_incorrect']:.3f}, n = {kb['n']} "
              f"-- the reference point for all flip metrics")
 
    rows = []
    for key, disp in order:
        e = m[key]
        p = e["flip_to_user_incorrect"]; n = e["n"]
        # Wald CI for a proportion: p +/- 1.96*sqrt(p(1-p)/n)
        hw = 1.96 * sqrt(p * (1 - p) / n) if n else 0.0
        rows.append([disp,
                     f"{p:.3f}",
                     f"[{p - hw:.3f}, {p + hw:.3f}]",
                     f"{e['flip_away_from_both_incorrect']:.3f}",
                     f"{e['accuracy_incorrect']:.3f}",
                     str(n),
                     f"{e.get('flip_to_user_correct', float('nan')):.3f}"])
    print_table("PER-CONDITION (incorrect placement; last col = control, "
                 "user-correct):",
                 ["Condition", "flip_to_user", "95% CI", "flip_away",
                  "accuracy", "n", "ctrl flip"],
                 rows)
 
    # baselines, descriptive
    print("\nBASELINES (descriptive, not in the model):")
    for key, disp in [("user_sycophancy_baseline", "user syc."),
                      ("user_sycophancy_bet_baseline", "bet (no money)")]:
        e = m[key]
        print(f"  {disp:18s} flip_to_user={e['flip_to_user_incorrect']:.3f}  "
              f"accuracy={e['accuracy_incorrect']:.3f}  n={e['n']}")
 
    print("\nPRECONDITION (baseline sycophancy, flip_to_user > 0):")
    print(f"  flip_to_user = {pc['flip_to_user_mean']:.3f}  "
          f"d = {pc['cohens_d']:.3f}  p_one-sided = {pc['p_one_sided']:.5e}")
 
    delta = (m["high_stakes"]["flip_to_user_incorrect"]
             - m["low_stakes"]["flip_to_user_incorrect"])
    print(f"\nSTAKES 0->2 IMPLIED CHANGE (high - low, flip-rate): {delta:+.3f}")
    print("  -- the cross-experiment comparable number for MMLU")
 
    if fit_results_csv:
        print_bstakes("mmlu", model, fit_results_csv,scale_note=" (log-odds)")
 
    labels = [d for _, d in order]
    means = [m[k]["flip_to_user_incorrect"] for k, _ in order]
    # Wald CI for a proportion: p +/- 1.96*sqrt(p(1-p)/n)
    import math
    lo, hi = [], []
    for k, _ in order:
        p = m[k]["flip_to_user_incorrect"]; n = m[k]["n"]
        hw = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0
        lo.append(p - hw); hi.append(p + hw)
    n_list = [m[k]["n"] for k, _ in order]
    plot(labels, means, lo, hi, "flip-to-user rate",
          f"MMLU: flip-to-user by stakes ({model})", None, plot_out, n_list=n_list)
    print("=" * 70)
 
 
# -----------------------------------------------------------------------------
# ELEPHANT
# -----------------------------------------------------------------------------
def summarize_elephant(results_csv, plot_out, model, fit_results_csv):
    import pandas as pd, math
    df = pd.read_csv(results_csv)
    elephant_metrics.check_raw_columns(df)   # clear error if CSV predates *_raw
    m = elephant_metrics.calculate_metrics(df)
    pc = elephant_metrics.precondition_baseline_sycophancy(df)
 
    order = [("baseline", "Baseline"), ("low_stakes", "Low"),
             ("medium_stakes", "Medium"), ("high_stakes", "High")]
 
    print("=" * 70)
    print("ELEPHANT MORAL-SYCOPHANCY -- PAPER SUMMARY")
    print("=" * 70)
    print("Primary metric: moral_sycophancy (NTA on BOTH perspectives). "
          "Stakes levels modelled; baseline descriptive.")
 
    rows = []
    for key, disp in order:
        e = m[key]
        p = e["moral_sycophancy"]; n = e["n"]
        hw = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0
        rows.append([disp, f"{p:.3f}", f"[{p - hw:.3f}, {p + hw:.3f}]", str(n),
                     str(e["n_dropped"])])
    print_table("PER-CONDITION (moral_sycophancy, 95% CI, n, "
                 "dropped):",
                 ["Condition", "moral_syc", "95% CI", "n", "dropped"], rows)
 
    print("\nPRECONDITION (baseline sycophancy, moral_sycophancy > 0):")
    print(f"  moral_sycophancy = {pc['moral_sycophancy_mean']:.3f}  "
          f"d = {pc['cohens_d']:.3f}  p_one-sided = {pc['p_one_sided']:.4f}")
 
    delta = (m["high_stakes"]["moral_sycophancy"]
             - m["low_stakes"]["moral_sycophancy"])
    print(f"\nSTAKES 0->2 IMPLIED CHANGE (high - low, moral_syc rate): "
          f"{delta:+.3f}")
    print("  -- the cross-experiment comparable number for ELEPHANT")
 
    if fit_results_csv:
        print_bstakes("elephant", model, fit_results_csv, scale_note=" (log-odds)")

    tm = elephant_metrics.additional_metrics(df)
    print("\nVERDICT TRANSITIONS (original_verdict -> flipped_verdict), "
          "rate [95% CI Wilson]:")
    print("  both_nta == moral_sycophancy; nta_to_yta == tracks the consensus flip")
    tlabels = [("both_yta", "Both YTA"), ("both_nta", "Both NTA"),
               ("yta_to_nta", "YTA->NTA"), ("nta_to_yta", "NTA->YTA"),
               ("refused", "Refused")]
    order_t = [("baseline", "Baseline"), ("low_stakes", "Low"),
               ("medium_stakes", "Medium"), ("high_stakes", "High")]
    header = ["Metric"] + [d for _, d in order_t]
    rows = []
    for key, disp in tlabels:
        row = [disp]
        for ckey, _ in order_t:
            cell = tm[ckey][key]
            lo, hi = cell["ci_95"]
            row.append(f"{cell['rate']*100:.2f}%  [{lo*100:.2f}, {hi*100:.2f}]")
        rows.append(row)
    print_table("", header, rows)
    # n note: 2x2 cells are over valid pairs; refused is over all pairs
    for ckey, disp in order_t:
        e = tm[ckey]
        print(f"  {disp:9s} n_total={e['n_total']}  n_valid={e['n_valid']}  "
              f"n_refused={e['n_refused']}")
 
    labels = [d for _, d in order]
    means = [m[k]["moral_sycophancy"] for k, _ in order]
    lo, hi = [], []
    for k, _ in order:
        p = m[k]["moral_sycophancy"]; n = m[k]["n"]
        hw = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0
        lo.append(p - hw); hi.append(p + hw)
    n_list = [m[k]["n"] for k, _ in order]
    plot(labels, means, lo, hi, "moral sycophancy rate",
          f"ELEPHANT: moral sycophancy by stakes ({model})", None, plot_out, n_list=n_list)
    print("=" * 70)
 
 
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    choices=["essay", "mmlu", "elephant"])
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--ground_truth_csv",
                    help="essay only -- the rubric reference grades")
    ap.add_argument("--plot_out", default=None,
                    help="PNG path for the condition-means plot")
    ap.add_argument("--model", required=True,
                    help="model name (e.g. 'gemini', 'gpt-4o', 'gpt-4o-mini')")
    ap.add_argument("--fit_results_csv", required=False, default=None,
                    help="fit results CSV file")
    args = ap.parse_args()
 
    if args.experiment == "essay":
        if not args.ground_truth_csv:
            ap.error("--ground_truth_csv is required for --experiment essay")
        summarize_essay(args.results_csv, args.ground_truth_csv, args.plot_out, args.model, args.fit_results_csv)
    elif args.experiment == "mmlu":
        summarize_mmlu(args.results_csv, args.plot_out, args.model, args.fit_results_csv)
    else:
        summarize_elephant(args.results_csv, args.plot_out, args.model, args.fit_results_csv)
 