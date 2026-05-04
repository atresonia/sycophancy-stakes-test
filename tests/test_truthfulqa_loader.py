"""Tests for data_loading.truthfulqa — TruthfulQA MC1 loader."""

import textwrap
from pathlib import Path

import pandas as pd

from data_loading.truthfulqa import load_truthfulqa


SYNTHETIC_CSV = textwrap.dedent("""\
    Type,Category,Question,Best Answer,Best Incorrect Answer,Correct Answers,Incorrect Answers,Source
    Adversarial,Misconceptions,What color is the sky?,Blue,Green,Blue,Green; Red; Yellow,wiki
    Adversarial,Science,What is 2+2?,4,5,4,5; 6; 7,wiki
    Adversarial,History,Who was first president?,Washington,Lincoln,Washington,Lincoln; Adams,wiki
    Adversarial,Geography,Capital of France?,Paris,London,Paris,London; Berlin; Rome,wiki
    Adversarial,Biology,How many legs does a dog have?,4,6,4,6; 8,wiki
    Adversarial,Math,What is pi?,3.14,4,3.14,4; 3; 2,wiki
    Adversarial,Language,What language is spoken in Japan?,Japanese,Chinese,Japanese,Chinese; Korean,wiki
    Adversarial,Physics,What falls faster in vacuum?,Same speed,Heavier one,Same speed,Heavier one; Lighter one,wiki
""")


def _write_csv(tmp_path: Path, content: str = SYNTHETIC_CSV) -> Path:
    csv_path = tmp_path / "TruthfulQA.csv"
    csv_path.write_text(content)
    return csv_path


def test_load_returns_expected_columns_and_dtypes(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    df = load_truthfulqa(csv_path)

    assert df.columns.tolist() == [
        "question_id", "question", "correct_answer", "incorrect_answer", "category",
    ]
    assert df.dtypes["question_id"] == "int64"


def test_filter_drops_empty_best_answer(tmp_path: Path) -> None:
    csv_with_empty = textwrap.dedent("""\
        Type,Category,Question,Best Answer,Best Incorrect Answer,Correct Answers,Incorrect Answers,Source
        Adversarial,Misconceptions,Q1?,Blue,Green,Blue,Green; Red,wiki
        Adversarial,Science,Q2?,,Bad,Good,Bad; Worse,wiki
        Adversarial,History,Q3?,Washington,Lincoln,Washington,Lincoln; Adams,wiki
    """)
    csv_path = _write_csv(tmp_path, csv_with_empty)
    df = load_truthfulqa(csv_path)

    # Row with empty Best Answer should be dropped
    assert len(df) == 2
    assert "Q2?" not in df["question"].values


def test_filter_drops_empty_incorrect_answers(tmp_path: Path) -> None:
    csv_with_empty = textwrap.dedent("""\
        Type,Category,Question,Best Answer,Best Incorrect Answer,Correct Answers,Incorrect Answers,Source
        Adversarial,Misconceptions,Q1?,Blue,Green,Blue,Green; Red,wiki
        Adversarial,Science,Q2?,Four,Bad,Four,,wiki
        Adversarial,History,Q3?,Washington,Lincoln,Washington,Lincoln; Adams,wiki
    """)
    csv_path = _write_csv(tmp_path, csv_with_empty)
    df = load_truthfulqa(csv_path)

    assert len(df) == 2
    assert "Q2?" not in df["question"].values


def test_num_ex_sampling_deterministic(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    df1 = load_truthfulqa(csv_path, num_ex=5, seed=42)
    df2 = load_truthfulqa(csv_path, num_ex=5, seed=42)

    assert len(df1) == 5
    assert df1["question_id"].tolist() == df2["question_id"].tolist()


def test_incorrect_answer_split(tmp_path: Path) -> None:
    csv_with_spaced = textwrap.dedent("""\
        Type,Category,Question,Best Answer,Best Incorrect Answer,Correct Answers,Incorrect Answers,Source
        Adversarial,Misconceptions,Q1?,Blue,Green,Blue,X ; Y ; Z,wiki
    """)
    csv_path = _write_csv(tmp_path, csv_with_spaced)
    df = load_truthfulqa(csv_path)

    assert df.iloc[0]["incorrect_answer"] == "X"
