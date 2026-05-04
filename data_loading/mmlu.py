"""Data loader for MMLU-Pro items.

Downloads TIGER-Lab/MMLU-Pro test split via the `datasets` library, caches as
parquet, and returns a clean DataFrame for downstream stakes experiments.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_mmlu_pro(
    path: Path | None = None,
    num_ex: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load MMLU-Pro items, downloading and caching to parquet on first use.

    Returns DataFrame with: question_id (sequential int), question, correct_answer,
    incorrect_answers (list[str] of all wrong options), answer_index, category,
    options (list[str] of all options). No single incorrect option is selected
    here — that happens downstream based on knowledge baseline results.
    """
    import pyarrow.parquet as pq

    path = path or Path("data/mmlu/mmlu_pro.parquet")
    if not path.exists():
        from datasets import load_dataset

        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(path)
    df = pd.DataFrame(pq.read_table(path).to_pylist())

    df = df.assign(
        question_id=range(len(df)),
        correct_answer=df.apply(lambda r: r["options"][r["answer_index"]], axis=1),
        incorrect_answers=df.apply(
            lambda r: [o for i, o in enumerate(r["options"]) if i != r["answer_index"]],
            axis=1,
        ),
    )[["question_id", "question", "correct_answer", "incorrect_answers",
       "answer_index", "category", "options"]]

    if num_ex is not None:
        df = df.sample(n=num_ex, random_state=seed)
    return df.reset_index(drop=True)


def select_hardest_distractor(
    knowledge_baseline_results: dict[int, dict[str, int]],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add 'incorrect_answer' column = the wrong option the model picked most often.

    Falls back to the first incorrect option if the model never picked a wrong
    answer for a question.
    """
    letters = "ABCDEFGHIJ"

    def pick(row: pd.Series) -> str:
        picks = knowledge_baseline_results.get(row["question_id"], {})
        correct = letters[row["answer_index"]]
        wrong = {l: c for l, c in picks.items() if l != correct and c > 0}
        if not wrong:
            return row["incorrect_answers"][0]
        return row["options"][letters.index(max(wrong, key=wrong.get))]

    return df.assign(incorrect_answer=df.apply(pick, axis=1))


def select_easy_distractor(
    knowledge_baseline_results: dict[int, dict[str, int]],
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add 'incorrect_answer_easy' column = a wrong option the model rarely picks.

    Candidates exclude the correct answer and (if present) the existing
    'incorrect_answer'. If any candidate has zero picks, samples uniformly from
    that zero set; otherwise picks argmin by count, ties broken by letter order.
    """
    letters = "ABCDEFGHIJ"
    has_hard = "incorrect_answer" in df.columns

    def pick(row: pd.Series) -> str:
        picks = knowledge_baseline_results.get(row["question_id"], {})
        correct = letters[row["answer_index"]]
        candidates = [
            letters[i] for i, opt in enumerate(row["options"])
            if letters[i] != correct
            and not (has_hard and opt == row["incorrect_answer"])
        ]
        zero = [l for l in candidates if picks.get(l, 0) == 0]
        if zero:
            chosen = str(rng.choice(zero))
        else:
            chosen = min(candidates, key=lambda l: (picks.get(l, 0), l))
        return row["options"][letters.index(chosen)]

    return df.assign(incorrect_answer_easy=df.apply(pick, axis=1))
