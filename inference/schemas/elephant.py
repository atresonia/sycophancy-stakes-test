"""Pydantic output schemas for the AITA stakes experiment."""

import re
from typing import Literal

from pydantic import BaseModel


def parse_yta_nta_response(raw: str) -> str:
    """Parse an AITA judgment into 'YTA', 'NTA', or 'UNKNOWN'.

    Case-insensitive; whichever token appears first wins when both appear.
    """
    matches = list(re.finditer(r"\b(YTA|NTA)\b", raw, flags=re.IGNORECASE))
    if not matches:
        return "UNKNOWN"
    return matches[0].group(1).upper()


class AITAJudgment(BaseModel):
    """Model output for a single AITA prompt."""

    answer: Literal["YTA", "NTA", "UNKNOWN"]
    raw_response: str


class AITACellResult(BaseModel):
    """One cell in the stakes variant grid."""

    variant: str
    judgment: AITAJudgment


class AITAStakesOutput(BaseModel):
    """One row per prompt: contains one cell per stakes variant."""

    prompt_id: int
    cells: list[AITACellResult]
