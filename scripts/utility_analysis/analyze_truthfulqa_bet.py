"""Analyze TruthfulQA bet-stakes experiment results.

Reads the .jsonl produced by ``generate_llm_resp_truthfulqa_bet.py``,
computes sycophancy rates per stakes level with bootstrap CIs, checks
positional bias, fits (alpha, beta, gamma) via MLE, prints summary tables,
and saves a bar chart.

Usage:
    python scripts/utility_analysis/analyze_truthfulqa_bet.py \
        --in_json data/truthfulqa/llm_resp_truthfulqa_bet_gemini.jsonl \
        --out_dir results/truthfulqa_bet \
        --n_bootstrap 1000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utility_analysis.utils.utility import (
    BET_CANDIDATES,
    TRUTHFULQA_STAKES_MAP,
    Observation,
    bet_accuracy,
    bet_harm,
    bet_validation,
    fit_utility,
    load_bet_jsonl_long,
)

# Ordered for display
VARIANT_ORDER = ["baseline", "low_stakes", "medium_stakes", "high_stakes"]
VARIANT_LABELS = {
    "baseline": "baseline",
    "low_stakes": "low_stakes  $10",
    "medium_stakes": "medium     $500",
    "high_stakes": "high     $5000",
}


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------


def compute_sycophancy_rates(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Compute per-variant sycophancy rate with 95% bootstrap CIs.

    Args:
        df: Long-form DataFrame from ``load_bet_jsonl_long``.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict keyed by variant name with keys: n, rate, ci_lo, ci_hi.
    """
    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, Any]] = {}

    for variant in VARIANT_ORDER:
        subset = df[df["variant"] == variant]
        n = len(subset)
        is_syc = subset["is_sycophantic"].values.astype(float)
        rate = float(is_syc.mean()) if n > 0 else 0.0

        # Bootstrap CI
        if n > 0:
            boot_rates = np.array([
                rng.choice(is_syc, size=n, replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            ci_lo = float(np.percentile(boot_rates, 2.5))
            ci_hi = float(np.percentile(boot_rates, 97.5))
        else:
            ci_lo, ci_hi = 0.0, 0.0

        results[variant] = {"n": n, "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi}

    return results


def compute_positional_rates(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute per-variant sycophancy rate separately by positional.

    Args:
        df: Long-form DataFrame from ``load_bet_jsonl_long``.

    Returns:
        Dict keyed by variant with keys: user_first_rate, friend_first_rate, delta.
    """
    results: dict[str, dict[str, Any]] = {}

    for variant in VARIANT_ORDER:
        subset = df[df["variant"] == variant]
        uf = subset[subset["positional"] == "user_first"]["is_sycophantic"]
        ff = subset[subset["positional"] == "friend_first"]["is_sycophantic"]

        uf_rate = float(uf.mean()) if len(uf) > 0 else 0.0
        ff_rate = float(ff.mean()) if len(ff) > 0 else 0.0

        results[variant] = {
            "user_first_rate": uf_rate,
            "friend_first_rate": ff_rate,
            "delta": abs(uf_rate - ff_rate),
        }

    return results


def build_observations(df: pd.DataFrame) -> list[Observation]:
    """Convert long-form DataFrame into Observations for MLE fitting.

    Excludes baseline (s=0). Ground truth is always 'friend' (user is always
    wrong in the primary design).

    Args:
        df: Long-form DataFrame from ``load_bet_jsonl_long``.

    Returns:
        List of Observation tuples.
    """
    obs: list[Observation] = []
    for _, row in df.iterrows():
        s = row["stakes_s"]
        if s == 0.0:  # exclude baseline
            continue
        obs.append(Observation(s, row["answer"], "friend"))
    return obs


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_rates_table(rates: dict[str, dict[str, Any]]) -> None:
    """Print sycophancy rate table to stdout."""
    print("\nPrimary: sycophancy rate (user always claims wrong)")
    print(f"{'variant':<22} | {'n':>5} | {'rate':>6} | 95% CI")
    print("-" * 60)
    for v in VARIANT_ORDER:
        r = rates[v]
        print(f"{VARIANT_LABELS[v]:<22} | {r['n']:>5} | {r['rate']:.2f}  | [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")


def print_positional_table(pos_rates: dict[str, dict[str, Any]]) -> None:
    """Print positional-bias check table to stdout."""
    print("\nPositional-bias check (should be ~equal within each variant)")
    print(f"{'variant':<22} | {'user_first rate':>16} | {'friend_first rate':>18} | {'delta':>6}")
    print("-" * 75)
    for v in VARIANT_ORDER:
        p = pos_rates[v]
        print(
            f"{VARIANT_LABELS[v]:<22} | {p['user_first_rate']:>16.2f} | "
            f"{p['friend_first_rate']:>18.2f} | {p['delta']:>6.2f}"
        )


def print_fit_result(fit: Any) -> None:
    """Print fitted utility parameters."""
    print("\nFitted utility parameters (baseline excluded):")
    print(f"  alpha = {fit.alpha:.2f}")
    print(f"  beta  = {fit.beta:.2f}")
    print(f"  gamma = {fit.gamma:.2f}    [hypothesis: gamma > 0]")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_bar_chart(
    rates: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    """Save bar chart of sycophancy rate with bootstrap CI error bars."""
    labels = [VARIANT_LABELS[v] for v in VARIANT_ORDER]
    means = [rates[v]["rate"] for v in VARIANT_ORDER]
    ci_lo = [rates[v]["ci_lo"] for v in VARIANT_ORDER]
    ci_hi = [rates[v]["ci_hi"] for v in VARIANT_ORDER]

    # Asymmetric error bars
    lower_err = [m - lo for m, lo in zip(means, ci_lo)]
    upper_err = [hi - m for m, hi in zip(means, ci_hi)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[lower_err, upper_err], capsize=5,
           color=["#9e9e9e", "#66bb6a", "#ffa726", "#ef5350"],
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Sycophancy rate P(answer = user)")
    ax.set_title("Sycophancy vs. Bet Stakes (TruthfulQA)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_analysis(
    in_json: Path,
    out_dir: Path,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full analysis pipeline.

    Args:
        in_json: Path to .jsonl from generate_llm_resp_truthfulqa_bet.py.
        out_dir: Directory for output (PNG).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Summary dict with keys: rates, positional, fit.
    """
    df = load_bet_jsonl_long(in_json)

    # Sycophancy rates with bootstrap CIs
    rates = compute_sycophancy_rates(df, n_bootstrap=n_bootstrap, seed=seed)
    print_rates_table(rates)

    # Positional bias check
    pos_rates = compute_positional_rates(df)
    print_positional_table(pos_rates)

    # Fit utility parameters (exclude baseline)
    obs = build_observations(df)
    fit = None
    if obs:
        fit = fit_utility(
            obs,
            candidates=BET_CANDIDATES,
            v_fn=bet_validation,
            a_fn=lambda r, _g: bet_accuracy(r),
            h_fn=lambda r, _g: bet_harm(r),
        )
        print_fit_result(fit)

    # Save bar chart
    out_dir.mkdir(parents=True, exist_ok=True)
    save_bar_chart(rates, out_dir / "sycophancy_vs_stakes.png")

    return {
        "rates": rates,
        "positional": pos_rates,
        "fit": fit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze TruthfulQA bet-stakes experiment results."
    )
    parser.add_argument("--in_json", type=Path, required=True,
                        help="Path to .jsonl from generate_llm_resp_truthfulqa_bet.py")
    parser.add_argument("--out_dir", type=Path, default=Path("results/truthfulqa_bet"),
                        help="Output directory for plots (default: results/truthfulqa_bet)")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    run_analysis(
        in_json=args.in_json,
        out_dir=args.out_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
