"""Pydantic output schemas for the essay grading pipeline."""

from typing import Literal
from pydantic import BaseModel, Field


class EssayGradeOutput(BaseModel):
    grade: Literal["A", "B", "C", "D", "F"]
    reasoning: str = Field(description="One sentence explaining why this grade was assigned.")


class EssayResponseOutput(BaseModel):
    """Output schema for LLM response for original essay prompt."""
    prompt: str
    grade: Literal["A", "B", "C", "D", "F"]


class EssayVariantsOutput(BaseModel):
    """Seven essay grades: 1 baseline + 3 low-stakes + 3 high-stakes framings.
    Each variant is graded in a separate LLM call."""
    baseline_response: EssayGradeOutput
    low_stakes_1_response: EssayGradeOutput
    low_stakes_2_response: EssayGradeOutput
    low_stakes_3_response: EssayGradeOutput
    high_stakes_1_response: EssayGradeOutput
    high_stakes_2_response: EssayGradeOutput
    high_stakes_3_response: EssayGradeOutput


class TimeStakesVariantsOutput(BaseModel):
    """Five essay grades for the 2×2 time × stakes experiment plus baseline.
    Each variant is graded in a separate LLM call."""
    baseline_response: EssayGradeOutput
    low_short_response: EssayGradeOutput
    low_long_response: EssayGradeOutput
    high_short_response: EssayGradeOutput
    high_long_response: EssayGradeOutput
