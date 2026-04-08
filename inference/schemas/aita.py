"""Legacy AITA pipeline output schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class OriginalStakes(BaseModel):
    """Original Prompt stakes rating (1-10) and brief justification."""
    prompt: str
    stakes_rating: int = Field(ge=1, le=10)
    justification: str


class Variant(BaseModel):
    """Variant (either high or low) for original prompt."""
    prompt: str
    stakes_rating: Optional[int] = Field(default=None, ge=1, le=10)
    diff_summary: str


class Variants(BaseModel):
    """High and low variants of original prompt"""
    low: Variant
    high: Variant


class StakesOutput(BaseModel):
    """Output schema for stakes generation task."""
    original: OriginalStakes
    variants: Variants
    # eval_prompt_template: str


class SycophancyEvalOutput(BaseModel):
    """Output schema for sycophancy evaluation."""
    low_stakes_score: int = Field(ge=0, le=100)
    high_stakes_score: int = Field(ge=0, le=100)
    reasoning: str
