"""Data loader for TruthfulQA MC1 items.

Reads TruthfulQA.csv (v0 format from https://github.com/sylinrl/TruthfulQA),
filters to rows with valid best answer and at least one incorrect answer,
and returns a clean DataFrame for downstream bet-stakes experiments.
"""

from pathlib import Path

import pandas as pd


def load_truthfulqa(
    path: Path,
    num_ex: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load TruthfulQA MC1 items, filtering to rows with valid answers.

    Args:
        path: Path to TruthfulQA.csv.
        num_ex: If set, deterministically sample this many rows.
        seed: Random seed for sampling.

    Returns:
        DataFrame with columns: question_id (int), question, correct_answer,
        incorrect_answer (first semicolon-delimited entry), category.
    """
    raise NotImplementedError
