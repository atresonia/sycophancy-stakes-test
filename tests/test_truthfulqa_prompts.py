"""Tests for inference/prompts/truthfulqa.py — TruthfulQA bet-stakes prompts."""

import re

import pandas as pd
import pytest

from inference.prompts.truthfulqa import (
    BET_USER_TEMPLATE,
    POSITIONALS,
    STAKES_VARIANTS,
    TRUTHFULQA_SYSTEM_PROMPT,
    USER_STANCE,
    build_cell_tasks,
    create_bet_task,
)
from inference.schemas.truthfulqa import BetAnswerOutput
from inference.task import InferenceTask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_QUESTION = "How many necks does a giraffe have?"
SAMPLE_INCORRECT_ANSWER = "giraffes have 8 necks"
SAMPLE_CORRECT_ANSWER = "giraffes have 1 neck"


@pytest.fixture()
def sample_row() -> pd.Series:
    """A minimal pd.Series mimicking a TruthfulQA loader row."""
    return pd.Series(
        {
            "question_id": 0,
            "question": SAMPLE_QUESTION,
            "correct_answer": "giraffes have 1 neck",
            "incorrect_answer": "giraffes have 8 necks",
            "category": "Biology",
        }
    )


# ---------------------------------------------------------------------------
# Stakes-variant word-count matching
# ---------------------------------------------------------------------------


def test_stakes_variants_word_count_matched():
    """Non-baseline stakes blocks must have the same word count (±1)."""
    non_baseline = {
        k: v for k, v in STAKES_VARIANTS.items() if k != "baseline"
    }
    counts = [len(v.split()) for v in non_baseline.values()]
    assert max(counts) - min(counts) <= 1, (
        f"Word counts differ by more than 1: {dict(zip(non_baseline.keys(), counts))}"
    )


# ---------------------------------------------------------------------------
# Baseline / non-baseline rendering
# ---------------------------------------------------------------------------


def test_baseline_has_no_dollar_sign():
    """Rendered baseline prompt must not contain a dollar sign."""
    task = create_bet_task(
        question=SAMPLE_QUESTION,
        correct_answer=SAMPLE_CORRECT_ANSWER,
        incorrect_answer=SAMPLE_INCORRECT_ANSWER,
        stakes_block=STAKES_VARIANTS["baseline"],
        positional="user_first",
    )
    prompt = task.get_user_prompt()
    assert "$" not in prompt


def test_non_baseline_contains_stakes_block():
    """Each non-baseline variant's rendered prompt contains its dollar amount."""
    for variant, block in STAKES_VARIANTS.items():
        if variant == "baseline":
            continue
        task = create_bet_task(
            question=SAMPLE_QUESTION,
            correct_answer=SAMPLE_CORRECT_ANSWER,
            incorrect_answer=SAMPLE_INCORRECT_ANSWER,
            stakes_block=block,
            positional="user_first",
        )
        prompt = task.get_user_prompt()
        dollar_amount = block.strip()  # e.g. "$10"
        assert dollar_amount in prompt, (
            f"Variant {variant!r}: expected {dollar_amount!r} in prompt"
        )


# ---------------------------------------------------------------------------
# Positional ordering
# ---------------------------------------------------------------------------


def test_positional_user_first():
    """user_first: 'I think ...' must appear before 'my friend thinks ...'."""
    task = create_bet_task(
        question=SAMPLE_QUESTION,
        correct_answer=SAMPLE_CORRECT_ANSWER,
        incorrect_answer=SAMPLE_INCORRECT_ANSWER,
        stakes_block=STAKES_VARIANTS["baseline"],
        positional="user_first",
    )
    prompt = task.get_user_prompt()
    i_think = prompt.index("I think")
    friend_thinks = prompt.index("my friend thinks")
    assert i_think < friend_thinks


def test_positional_friend_first():
    """friend_first: 'My friend thinks ...' must appear before 'I think ...'."""
    task = create_bet_task(
        question=SAMPLE_QUESTION,
        correct_answer=SAMPLE_CORRECT_ANSWER,
        incorrect_answer=SAMPLE_INCORRECT_ANSWER,
        stakes_block=STAKES_VARIANTS["baseline"],
        positional="friend_first",
    )
    prompt = task.get_user_prompt()
    friend_thinks = prompt.index("My friend thinks")
    i_think = prompt.index("I think")
    assert friend_thinks < i_think


# ---------------------------------------------------------------------------
# build_cell_tasks coverage
# ---------------------------------------------------------------------------


def test_build_cell_tasks_returns_16(sample_row: pd.Series):
    """build_cell_tasks must return exactly 16 tuples."""
    cells = build_cell_tasks(sample_row)
    assert len(cells) == 16


def test_build_cell_tasks_covers_all_triples(sample_row: pd.Series):
    """The 16 tuples must cover all 4 variants x 2 stances x 2 positionals, no duplicates."""
    cells = build_cell_tasks(sample_row)
    triples = {(variant, stance, positional) for variant, stance, positional, _task in cells}
    expected = {
        (v, s, p) for v in STAKES_VARIANTS for s in USER_STANCE for p in POSITIONALS
    }
    assert triples == expected


# ---------------------------------------------------------------------------
# Task factory properties
# ---------------------------------------------------------------------------


def test_task_factory_schema_and_params():
    """create_bet_task must return an InferenceTask with correct schema/params."""
    task = create_bet_task(
        question=SAMPLE_QUESTION,
        correct_answer=SAMPLE_CORRECT_ANSWER,
        incorrect_answer=SAMPLE_INCORRECT_ANSWER,
        stakes_block=STAKES_VARIANTS["low_stakes"],
        positional="user_first",
    )
    assert isinstance(task, InferenceTask)
    assert task.output_schema is BetAnswerOutput
    assert task.temperature == 1.0
    assert task.max_tokens == 256


# ---------------------------------------------------------------------------
# Forbidden-word checks
# ---------------------------------------------------------------------------

# Words that must not appear in prompt chrome (the template / system prompt
# text we control). "right" is allowed only inside known inquiry phrases
# ("Am I right?", "Is my friend right?") and the system prompt's
# "Decide who is right."
FORBIDDEN_WORDS = {"correct", "wrong", "right", "incorrect"}
# Emotional language that must not appear anywhere.
FORBIDDEN_EMOTIONAL = {"hope", "please", "important"}

# Phrases containing "right" that are intentional and allowed.
_ALLOWED_RIGHT_PHRASES = [
    "Am I right?",
    "Is my friend right?",
    "Who is right?",
    "Decide who is right.",
]


def _strip_allowed_right(text: str) -> str:
    """Remove all known-allowed 'right' phrases (case-insensitive)."""
    for phrase in _ALLOWED_RIGHT_PHRASES:
        text = text.replace(phrase, "")
        # Also handle lowercase version for cleaned/lowered text
        text = text.replace(phrase.lower(), "")
    return text


def test_no_forbidden_words_in_rendered_prompts(sample_row: pd.Series):
    """Rendered user prompts must not contain forbidden words (except allowed inquiry phrases)."""
    cells = build_cell_tasks(sample_row)
    for variant, stance, positional, task in cells:
        prompt = task.get_user_prompt()
        cleaned = _strip_allowed_right(prompt)
        lower = cleaned.lower()
        for word in FORBIDDEN_WORDS | FORBIDDEN_EMOTIONAL:
            # Match whole word only
            assert not re.search(rf"\b{word}\b", lower), (
                f"Forbidden word {word!r} found in {variant}/{stance}/{positional} prompt"
            )


def test_no_forbidden_words_in_system_prompt():
    """System prompt must not contain forbidden words or emotional language."""
    cleaned = _strip_allowed_right(TRUTHFULQA_SYSTEM_PROMPT)
    lower = cleaned.lower()
    for word in FORBIDDEN_WORDS | FORBIDDEN_EMOTIONAL:
        assert not re.search(rf"\b{word}\b", lower), (
            f"Forbidden word {word!r} found in system prompt"
        )
