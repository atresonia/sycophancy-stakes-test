"""Tests for inference.schemas.truthfulqa — Pydantic schemas for bet-stakes experiment."""

import pytest
from pydantic import ValidationError

from inference.schemas.truthfulqa import (
    BetAnswerOutput,
    BetCellResult,
    BetVariantsOutput,
)

VARIANTS = ("baseline", "low_stakes", "medium_stakes", "high_stakes")
POSITIONALS = ("user_first", "friend_first")


def _make_cell(variant: str, positional: str) -> BetCellResult:
    return BetCellResult(
        variant=variant,
        positional=positional,
        response=BetAnswerOutput(answer="friend", reasoning="Factually correct."),
    )


def _make_8_cells() -> list[BetCellResult]:
    return [
        _make_cell(v, p)
        for v in VARIANTS
        for p in POSITIONALS
    ]


def test_bet_variants_output_roundtrip() -> None:
    cells = _make_8_cells()
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
    assert len(restored.cells) == 8
    assert restored.cells[0].variant == "baseline"
    assert restored.cells[0].response.answer == "friend"


def test_bet_variants_output_rejects_too_few_cells() -> None:
    cells = _make_8_cells()[:7]
    with pytest.raises(ValidationError, match="Expected exactly 8 cells"):
        BetVariantsOutput(
            question_id=0,
            correct_answer="Blue",
            incorrect_answer="Green",
            repeat_idx=0,
            cells=cells,
        )


def test_bet_variants_output_rejects_too_many_cells() -> None:
    cells = _make_8_cells() + [_make_cell("baseline", "user_first")]
    with pytest.raises(ValidationError, match="Expected exactly 8 cells"):
        BetVariantsOutput(
            question_id=0,
            correct_answer="Blue",
            incorrect_answer="Green",
            repeat_idx=0,
            cells=cells,
        )
