"""Pydantic output schemas for the TruthfulQA bet-stakes experiment."""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class BetAnswerOutput(BaseModel):
    """Model output for a single bet question."""

    answer: Literal["user", "friend"]
    reasoning: str = Field(description="1-3 sentences explaining the choice.")


class BetCellResult(BaseModel):
    """One cell in the variant x positional grid."""

    variant: Literal["baseline", "low_stakes", "medium_stakes", "high_stakes"]
    positional: Literal["user_first", "friend_first"]
    response: BetAnswerOutput


class BetVariantsOutput(BaseModel):
    """One row per question x repeat: contains all 8 cells (4 variants x 2 positionals)."""

    question_id: int
    correct_answer: str
    incorrect_answer: str
    repeat_idx: int
    cells: list[BetCellResult]

    @model_validator(mode="after")
    def _check_cell_count(self) -> Self:
        """Enforce exactly 8 cells (4 variants x 2 positionals)."""
        if len(self.cells) != 8:
            raise ValueError(
                f"Expected exactly 8 cells, got {len(self.cells)}"
            )
        return self
