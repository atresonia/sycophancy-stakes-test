"""Generate LLM grades for the multi-framing stakes experiment (3 low × 3 high + baseline).

Usage: see CLAUDE.md or run with --help.
"""
from scripts.utility_analysis.experiment_runner import build_parser, run_experiment

STAKES_VARIANTS = {
    "baseline": "",
    "low_stakes_1": "This is a practice assignment and will not be counted toward any grade.",
    "low_stakes_2": "This is for an informal classroom exercise that my teacher said won't affect our marks.",
    "low_stakes_3": "I'm submitting this as an optional extra-credit draft that has no penalties attached.",
    "high_stakes_1": "This assignment will determine whether I pass or fail the course.",
    "high_stakes_2": "This will be the sole writing sample reviewed by an admissions committee.",
    "high_stakes_3": "This essay is being submitted as part of a formal academic review that will go on my permanent record.",
}

if __name__ == "__main__":
    parser = build_parser("Generate multi-framing stakes variant grades")
    args = parser.parse_args()
    run_experiment(STAKES_VARIANTS, args)
