"""Tests for inference.schemas.truthfulqa — Pydantic schemas for bet-stakes experiment."""

import pytest
from pydantic import ValidationError

from inference.schemas.truthfulqa import (
    BetAnswerOutput,
    BetCellResult,
    BetVariantsOutput,
    parse_mc_response,
    parse_one_word_response,
)

VARIANTS = ("baseline", "low_stakes", "medium_stakes", "high_stakes")
POSITIONALS = ("user_first", "friend_first")
STANCES = ("user_correct", "user_incorrect")


def _make_cell(variant: str, positional: str, stance: str = "user_incorrect") -> BetCellResult:
    return BetCellResult(
        variant=variant,
        positional=positional,
        user_stance=stance,
        response=BetAnswerOutput(answer="friend", raw_response="Friend"),
    )


def _make_16_cells() -> list[BetCellResult]:
    return [
        _make_cell(v, p, s)
        for v in VARIANTS
        for s in STANCES
        for p in POSITIONALS
    ]


def test_bet_variants_output_roundtrip() -> None:
    cells = _make_16_cells()
    obj = BetVariantsOutput(
        question_id=0,
        correct_answer="Blue",
        incorrect_answer="Green",
        repeat_idx=0,
        cells=cells,
    )
    json_str = obj.model_dump_json()
    restored = BetVariantsOutput.model_validate_json(json_str)

    assert restored.question_id == obj.question_id
    assert len(restored.cells) == 16
    assert restored.cells[0].variant == "baseline"
    assert restored.cells[0].response.answer == "friend"


def test_bet_variants_output_rejects_too_few_cells() -> None:
    cells = _make_16_cells()[:15]
    with pytest.raises(ValidationError, match="Expected exactly 16 cells"):
        BetVariantsOutput(
            question_id=0,
            correct_answer="Blue",
            incorrect_answer="Green",
            repeat_idx=0,
            cells=cells,
        )


def test_bet_variants_output_rejects_too_many_cells() -> None:
    cells = _make_16_cells() + [_make_cell("baseline", "user_first")]
    with pytest.raises(ValidationError, match="Expected exactly 16 cells"):
        BetVariantsOutput(
            question_id=0,
            correct_answer="Blue",
            incorrect_answer="Green",
            repeat_idx=0,
            cells=cells,
        )


# ---------------------------------------------------------------------------
# Parser tests (migrated from __main__ block in schemas/truthfulqa.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("You", "user"),
        ("Friend", "friend"),
        ("you.", "user"),
        (" FRIEND ", "friend"),
        ("You are right", "user"),
        ("I think Friend is correct", "friend"),
        ("I cannot determine", "unparseable"),
        ("You\n", "user"),
    ],
)
def test_parse_one_word_response(raw: str, expected: str) -> None:
    assert parse_one_word_response(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("A", "A"),
        ("B", "B"),
        ("a", "A"),
        (" b ", "B"),
        ("(A)", "A"),
        ("A.", "A"),
        ("The answer is B", "B"),
        ("I'm not sure", "unparseable"),
    ],
)
def test_parse_mc_response(raw: str, expected: str) -> None:
    assert parse_mc_response(raw) == expected
