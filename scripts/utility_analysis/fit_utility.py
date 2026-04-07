"""
Script to fit the utility function via MLE to the data (model-specific)

Ground truth for sycophancy is the LLM's own grade to the base essay text
(no stakes/user framing). Use the --gt_field human to switch to human grader gt.

Fitting using low_stakes (s=1) and high_stakes (s=3) variants only 
(baseline is excluded).

Usage:
    llm gt: python scripts/utility_analysis/fit_utility.py \\
        --in_json gemini:data/essay_grading/llm_resp_stakes_gemini_400_500.jsonl \\
                  openai:data/essay_grading/llm_resp_stakes_openai_400_500.jsonl \\
                  claude:data/essay_grading/llm_resp_stakes_claude_400_500.jsonl
    human gt: python scripts/utility_analysis/fit_utility.py --gt_field human \\
        --in_json gemini:data/essay_grading/llm_resp_stakes_gemini_400_500.jsonl \\
                  openai:data/essay_grading/llm_resp_stakes_openai_400_500.jsonl \\
                  claude:data/essay_grading/llm_resp_stakes_claude_400_500.jsonl
Note: This script is mainly for the essay grading task. 
We will re-evaluate this for other tasks (ex: AITA).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

# make utils importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.utility import (
    GRADES, fit_mle, fit_mle_null_hypothesis, log_likelihood, choice_probs, FitResult, Observation
)

STAKES_MAP = {
    "low_stakes":  1.0,
    "high_stakes": 3.0,
}

def load_observations(in_json: str, gt_field: str = "llm_response") -> list[Observation]:
    """Extract (s, observed_grade, ground_truth) from the input JSONL file.

    gt_field: 'llm' (LLM's grade - no stakes) or 'human' (human grader grade)
    Returns:
        list of observations (stakes_level, observed_grade, ground_truth)
    """
    observations = []
    skipped = 0
    with open(in_json) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            gt = rec.get("llm_response" if gt_field == "llm" else "ground_truth", "")
            if gt not in GRADES:
                skipped += 1
                continue
            for s, label in STAKES_MAP.items():
                output = rec.get(f"{s}_output")
                if output is None or output.get("grade") not in GRADES:
                    skipped += 1
                    continue
                observations.append(Observation(stakes_level=label, observed_grade=output["grade"], ground_truth=gt))
    if skipped > 0:
        print(f"WARNING: Skipped {skipped} records due to invalid/missing grades", file=sys.stderr)
    return observations

def lrt_p_value(full: FitResult, null: FitResult) -> float:
    """Likelihood ratio test p-value for the null hypothesis (γ=0).
    We use the chi-squared distribution to test the difference between the full and null models.
    """
    stat = 2.0 * (full.log_likelihood - null.log_likelihood)
    return chi2.sf(max(0.0, stat), df=1)


def print_results(label: str, full: FitResult, null: FitResult) -> None:
    p = lrt_p_value(full, null)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"\n{'─' * 52}")
    print(f"  Model : {label}")
    print(f"  n obs : {full.n_obs}  ({full.n_obs // 2} essays × 2 stakes levels)")
    print(f"{'─' * 52}")
    print(f"  α (validation weight)  : {full.alpha:+.4f}")
    print(f"  β (accuracy weight)    : {full.beta:+.4f}")
    print(f"  γ (harm/stakes weight) : {full.gamma:+.4f}")
    print(f"  LL full model          : {full.log_likelihood:.3f}")
    print(f"  LL null (γ=0)          : {null.log_likelihood:.3f}")
    print(f"  LRT p-value (γ≠0)      : {p:.4f}  {stars}")
    if not full.converged:
        print("  [!] optimizer did not converge — results may be unreliable")
    print(f"{'─' * 52}")


def print_choice_table(label: str, result: FitResult, gt: str = "C") -> None:
    """Print model-implied choice probabilities for a given ground truth grade."""
    print(f"\n  Implied P(grade | s, gt={gt}) for {label}:")
    print(f"  {'Grade':<6} {'s=1 (low)':>12} {'s=3 (high)':>12}")
    for r in GRADES:
        p_low  = choice_probs(1.0, gt, result.alpha, result.beta, result.gamma)[r]
        p_high = choice_probs(3.0, gt, result.alpha, result.beta, result.gamma)[r]
        print(f"  {r:<6} {p_low:>12.3f} {p_high:>12.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_json", nargs="+", required=True, metavar="LABEL:PATH",
        help="One or more label:path pairs.",
    )
    parser.add_argument(
        "--gt_field", choices=["llm", "human"], default="llm",
        help="Ground truth source: 'llm' = llm_response (default), 'human' = ground_truth.",
    )
    parser.add_argument(
        "--choice_table_gt", default="C",
        help="Ground truth grade to use when printing implied choice probability table (default: C).",
    )
    parser.add_argument("--n_restarts", type=int, default=8)
    args = parser.parse_args()

    gt_label = "llm_response" if args.gt_field == "llm" else "human_grader"
    print(f"\nGround truth: {gt_label}")

    all_results: dict[str, tuple[FitResult, FitResult]] = {}

    for entry in args.in_json:
        label, _, path_str = entry.partition(":")
        path = Path(path_str)
        if not path.exists():
            print(f"[error] file not found for '{label}': {path}", file=sys.stderr)
            continue

        print(f"\nFitting {label} ...", file=sys.stderr)
        obs = load_observations(path, args.gt_field)
        if not obs:
            print(f"[error] no valid observations in {path}", file=sys.stderr)
            continue

        full = fit_mle(obs, n_restarts=args.n_restarts)
        null = fit_mle_null_hypothesis(obs)
        all_results[label] = (full, null)

        print_results(label, full, null)
        print_choice_table(label, full, gt=args.choice_table_gt)

    if len(all_results) > 1:
        print(f"\n\n{'═' * 52}")
        print("  Summary across models")
        print(f"{'═' * 52}")
        print(f"  {'Model':<10} {'α':>8} {'β':>8} {'γ':>8} {'p(γ≠0)':>10}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for label, (full, null) in all_results.items():
            p = lrt_p_value(full, null)
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {label:<10} {full.alpha:>+8.4f} {full.beta:>+8.4f} {full.gamma:>+8.4f} {p:>8.4f} {stars}")
        print(f"{'═' * 52}")

if __name__ == "__main__":
    main()
