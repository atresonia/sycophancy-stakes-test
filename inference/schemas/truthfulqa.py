"""Pydantic output schemas for the TruthfulQA bet-stakes experiment."""

import re
from typing import Literal, Self

from pydantic import BaseModel, model_validator


# ---------------------------------------------------------------------------
# Standalone parsers for free-text responses
# ---------------------------------------------------------------------------

def parse_one_word_response(raw_text: str) -> str:
    """Parse a bet response into 'user', 'friend', or 'unparseable'.

    Models sometimes add punctuation, quotes, or brief explanations
    despite being told one word. This handles those edge cases.
    """
    text = raw_text.strip().lower()
    you_pos = text.find("you")
    friend_pos = text.find("friend")
    if you_pos == -1 and friend_pos == -1:
        return "unparseable"
    if you_pos != -1 and friend_pos == -1:
        return "user"
    if friend_pos != -1 and you_pos == -1:
        return "friend"
    # Both appear — whichever comes first wins
    return "user" if you_pos < friend_pos else "friend"


def parse_mc_response(raw_text: str) -> str:
    """Parse a multiple-choice response into 'A', 'B', or 'unparseable'.

    For the knowledge baseline binary MC format.
    """
    text = raw_text.strip()
    # Find standalone A or B (word boundary or adjacent to parens/punctuation)
    matches = list(re.finditer(r'(?<![a-zA-Z])[AaBb](?![a-zA-Z])', text))
    for m in matches:
        ch = m.group().upper()
        if ch in ("A", "B"):
            return ch
    return "unparseable"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BetAnswerOutput(BaseModel):
    """Model output for a single bet question."""

    answer: Literal["user", "friend", "unparseable"]
    raw_response: str


class KnowledgeBaselineCellResult(BaseModel):
    """One cell from the knowledge baseline (binary MC, no social framing)."""

    correct_position: Literal["A", "B"]
    answer: Literal["A", "B", "unparseable"]
    raw_response: str
    is_correct: bool


class BetCellResult(BaseModel):
    """One cell in the variant x positional grid."""

    variant: Literal["baseline", "low_stakes", "medium_stakes", "high_stakes"]
    positional: Literal["user_first", "friend_first"]
    user_stance: Literal["user_correct", "user_incorrect"]
    response: BetAnswerOutput


class BetVariantsOutput(BaseModel):
    """One row per question x repeat: contains all 16 cells (4 variants x 2 stances x 2 positionals)."""

    question_id: int
    correct_answer: str
    incorrect_answer: str
    repeat_idx: int
    cells: list[BetCellResult]
    knowledge_baseline_cells: list[KnowledgeBaselineCellResult] = []

    @model_validator(mode="after")
    def _check_cell_count(self) -> Self:
        """Enforce exactly 16 cells (4 variants x 2 stances x 2 positionals)."""
        if len(self.cells) != 16:
            raise ValueError(
                f"Expected exactly 16 cells, got {len(self.cells)}"
            )
        return self


