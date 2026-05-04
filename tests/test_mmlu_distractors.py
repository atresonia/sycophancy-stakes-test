"""Tests for data_loading.mmlu.select_easy_distractor."""

import numpy as np
import pandas as pd

from data_loading.mmlu import select_easy_distractor


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal mmlu-shaped DataFrame from per-row dicts."""
    full = []
    for r in rows:
        opts = r["options"]
        ai = r["answer_index"]
        record = {
            "question_id": r["question_id"],
            "question": r.get("question", "?"),
            "correct_answer": opts[ai],
            "incorrect_answers": [o for i, o in enumerate(opts) if i != ai],
            "answer_index": ai,
            "category": "test",
            "options": opts,
        }
        if "incorrect_answer" in r:
            record["incorrect_answer"] = r["incorrect_answer"]
        full.append(record)
    return pd.DataFrame(full)


def test_all_zero_easy_sampled_from_zero_set_and_not_hard() -> None:
    df = _make_df([
        {"question_id": 1, "options": ["a", "b", "c", "d"], "answer_index": 0,
         "incorrect_answer": "b"},
    ])
    kb: dict[int, dict[str, int]] = {1: {}}  # missing letters treated as 0

    # Run many trials with a fresh seed each time; easy must always come from
    # the zero set {C, D} and never equal the hard distractor "b".
    seen = set()
    for seed in range(50):
        rng = np.random.default_rng(seed)
        out = select_easy_distractor(kb, df, rng)
        easy = out.loc[0, "incorrect_answer_easy"]
        assert easy in {"c", "d"}
        assert easy != out.loc[0, "incorrect_answer"]
        seen.add(easy)
    assert seen == {"c", "d"}, "rng should eventually sample both zero options"


def test_mixed_case_argmin_when_no_zero_candidates() -> None:
    df = _make_df([
        {"question_id": 1, "options": ["a", "b", "c", "d"], "answer_index": 0,
         "incorrect_answer": "b"},
    ])
    # All wrong options have nonzero picks; argmin among {C, D} is D (count 1).
    kb = {1: {"B": 5, "C": 3, "D": 1}}
    out = select_easy_distractor(kb, df, np.random.default_rng(0))
    assert out.loc[0, "incorrect_answer_easy"] == "d"
    assert out.loc[0, "incorrect_answer_easy"] != out.loc[0, "incorrect_answer"]


def test_mixed_case_argmin_tie_break_by_letter_order() -> None:
    df = _make_df([
        {"question_id": 1, "options": ["a", "b", "c", "d"], "answer_index": 0,
         "incorrect_answer": "b"},
    ])
    # Candidates {C, D} tie at count 2; letter order picks C.
    kb = {1: {"B": 3, "C": 2, "D": 2}}
    out = select_easy_distractor(kb, df, np.random.default_rng(0))
    assert out.loc[0, "incorrect_answer_easy"] == "c"


def test_easy_never_equals_hard_across_rows() -> None:
    df = _make_df([
        {"question_id": 0, "options": ["a", "b", "c", "d"], "answer_index": 0,
         "incorrect_answer": "b"},
        {"question_id": 1, "options": ["a", "b", "c", "d"], "answer_index": 0,
         "incorrect_answer": "c"},
        {"question_id": 2, "options": ["a", "b", "c", "d"], "answer_index": 0,
         "incorrect_answer": "d"},
    ])
    kb = {
        0: {"B": 5, "C": 1, "D": 0},   # zero-set picks D
        1: {"B": 2, "C": 7, "D": 3},   # argmin among {B, D} → B
        2: {"B": 1, "C": 4, "D": 9},   # argmin among {B, C} → B
    }
    out = select_easy_distractor(kb, df, np.random.default_rng(0))
    for i in range(len(df)):
        assert out.loc[i, "incorrect_answer_easy"] != out.loc[i, "incorrect_answer"]


def test_determinism_under_fixed_seed() -> None:
    df = _make_df([
        {"question_id": 1, "options": ["a", "b", "c", "d"], "answer_index": 0,
         "incorrect_answer": "b"},
        {"question_id": 2, "options": ["w", "x", "y", "z"], "answer_index": 1,
         "incorrect_answer": "w"},
    ])
    kb = {1: {}, 2: {}}  # all-zero → exercises rng.choice for both rows
    out1 = select_easy_distractor(kb, df, np.random.default_rng(7))
    out2 = select_easy_distractor(kb, df, np.random.default_rng(7))
    assert out1["incorrect_answer_easy"].tolist() == out2["incorrect_answer_easy"].tolist()
