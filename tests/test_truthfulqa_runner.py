"""Tests for the TruthfulQA bet-stakes experiment runner.

All tests mock ``LLMClient.run`` — no real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock

import orjson
import pandas as pd
import pytest

from inference.batch_inference import RunMode, load_latest_records, run_multi_task_inference
from inference.client import EndpointConfig, LLMClient, stable_key
from inference.prompts.truthfulqa import KNOWLEDGE_POSITIONS, POSITIONALS, STAKES_VARIANTS, USER_STANCE
from inference.schemas.truthfulqa import BetAnswerOutput, BetVariantsOutput
from inference.task import InferenceTask
from scripts.utility_analysis.generate_llm_resp_truthfulqa_bet import (
    build_truthfulqa_task_factories,
    expand_with_repeats,
    extra_fields_fn,
    make_combine_outputs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = pd.DataFrame(
    {
        "question_id": [0, 1, 2],
        "question": [
            "How many moons does Earth have?",
            "What is the capital of France?",
            "How many legs does a spider have?",
        ],
        "correct_answer": ["1", "Paris", "8"],
        "incorrect_answer": ["7", "London", "6"],
        "category": ["Science", "Geography", "Biology"],
    }
)


@pytest.fixture()
def sample_questions_df() -> pd.DataFrame:
    return SAMPLE_QUESTIONS.copy()


@pytest.fixture()
def mock_bet_client() -> LLMClient:
    """LLMClient that returns a deterministic BetAnswerOutput."""
    client = LLMClient(
        endpoint=EndpointConfig(provider="openai_compat", model="test-bet", api_key="x"),
        cache_dir=None,
    )

    async def fake_run(task: InferenceTask) -> Dict[str, Any]:
        # Bet tasks return raw text (output_schema=None); knowledge baseline too.
        return {
            "response_content": "Friend",
            "raw_response": {},
        }

    client.run = AsyncMock(side_effect=fake_run)
    return client


# ---------------------------------------------------------------------------
# expand_with_repeats
# ---------------------------------------------------------------------------


def test_expand_with_repeats_shape(sample_questions_df: pd.DataFrame) -> None:
    """3 questions x 2 repeats = 6 rows with unique indices."""
    expanded = expand_with_repeats(sample_questions_df, m_repeats=2)
    assert len(expanded) == 6
    assert expanded.index.is_unique


def test_expand_with_repeats_index_encoding(sample_questions_df: pd.DataFrame) -> None:
    """Index encodes question_id * m_repeats + repeat_idx and is reversible."""
    m = 5
    expanded = expand_with_repeats(sample_questions_df, m_repeats=m)
    for _, row in expanded.iterrows():
        qid = int(row["question_id"])
        ridx = int(row["repeat_idx"])
        expected_index = qid * m + ridx
        assert expected_index in expanded.index


def test_expand_with_repeats_all_pairs(sample_questions_df: pd.DataFrame) -> None:
    """Every (question_id, repeat_idx) pair appears exactly once."""
    m = 2
    expanded = expand_with_repeats(sample_questions_df, m_repeats=m)
    pairs = set(zip(expanded["question_id"], expanded["repeat_idx"]))
    expected = {(qid, r) for qid in [0, 1, 2] for r in range(m)}
    assert pairs == expected


# ---------------------------------------------------------------------------
# build_truthfulqa_task_factories
# ---------------------------------------------------------------------------


def test_build_task_factories_returns_18_keys() -> None:
    """Must return 18 keys: 2 knowledge baseline + 16 bet."""
    factories = build_truthfulqa_task_factories(STAKES_VARIANTS)
    assert len(factories) == 18
    expected_bet_keys = {
        f"{v}_{s}_{p}" for v in STAKES_VARIANTS for s in USER_STANCE for p in POSITIONALS
    }
    expected_kb_keys = {f"knowledge_baseline_correct_{pos}" for pos in KNOWLEDGE_POSITIONS}
    assert set(factories.keys()) == expected_bet_keys | expected_kb_keys


def test_task_factory_sets_repeat_idx() -> None:
    """Each factory must set repeat_idx on the returned InferenceTask."""
    factories = build_truthfulqa_task_factories(STAKES_VARIANTS)
    row = pd.Series(
        {
            "question_id": 0,
            "question": "What color is the sky?",
            "correct_answer": "blue",
            "incorrect_answer": "green",
            "repeat_idx": 7,
        }
    )
    for key, factory in factories.items():
        task = factory(row)
        assert isinstance(task, InferenceTask)
        assert task.repeat_idx == 7, f"Factory {key!r} did not set repeat_idx"


# ---------------------------------------------------------------------------
# extra_fields_fn
# ---------------------------------------------------------------------------


def test_extra_fields_fn_extracts_correct_fields() -> None:
    row = pd.Series(
        {
            "question_id": 42,
            "question": "ignored",
            "correct_answer": "Paris",
            "incorrect_answer": "London",
            "repeat_idx": 3,
            "category": "Geography",
        }
    )
    fields = extra_fields_fn(row)
    assert fields == {
        "question_id": 42,
        "correct_answer": "Paris",
        "incorrect_answer": "London",
        "repeat_idx": 3,
    }


# ---------------------------------------------------------------------------
# make_combine_outputs
# ---------------------------------------------------------------------------


def _make_mock_outputs() -> Dict[str, Any]:
    """Build a dict mimicking run_multi_task_inference's per-key outputs (raw text)."""
    outputs: Dict[str, Any] = {}
    for pos in KNOWLEDGE_POSITIONS:
        outputs[f"knowledge_baseline_correct_{pos}_output"] = "A"
    for variant in STAKES_VARIANTS:
        for stance in USER_STANCE:
            for positional in POSITIONALS:
                key = f"{variant}_{stance}_{positional}_output"
                outputs[key] = "Friend"
    return outputs


def _make_mock_base_record() -> Dict[str, Any]:
    return {
        "question_id": 5,
        "correct_answer": "Paris",
        "incorrect_answer": "London",
        "repeat_idx": 2,
    }


def test_combine_outputs_produces_valid_output() -> None:
    """combine_outputs must produce a BetVariantsOutput-valid dict with 16 bet + 2 kb cells."""
    combine = make_combine_outputs(STAKES_VARIANTS)
    result = combine(_make_mock_outputs(), _make_mock_base_record())
    assert "output" in result
    # Validate via Pydantic
    bvo = BetVariantsOutput.model_validate(result["output"])
    assert len(bvo.cells) == 16
    assert len(bvo.knowledge_baseline_cells) == 2
    assert bvo.question_id == 5
    assert bvo.repeat_idx == 2


def test_combine_outputs_covers_all_cells() -> None:
    """All 16 (variant, stance, positional) triples must appear in bet cells."""
    combine = make_combine_outputs(STAKES_VARIANTS)
    result = combine(_make_mock_outputs(), _make_mock_base_record())
    bvo = BetVariantsOutput.model_validate(result["output"])
    triples = {(c.variant, c.user_stance, c.positional) for c in bvo.cells}
    expected = {(v, s, p) for v in STAKES_VARIANTS for s in USER_STANCE for p in POSITIONALS}
    assert triples == expected


# ---------------------------------------------------------------------------
# Cache key distinctness across repeats
# ---------------------------------------------------------------------------


def test_cache_key_distinctness() -> None:
    """Two tasks differing only in repeat_idx must produce different cache keys."""
    task_a = InferenceTask(
        user_prompt="What is 2+2?",
        system_prompt="Answer.",
        repeat_idx=0,
    )
    task_b = InferenceTask(
        user_prompt="What is 2+2?",
        system_prompt="Answer.",
        repeat_idx=1,
    )

    def make_payload(t: InferenceTask) -> Dict[str, Any]:
        payload = {
            "model": "test",
            "user_prompt": t.get_user_prompt(),
            "system_prompt": t.system_prompt,
            "temperature": t.temperature,
            "max_tokens": t.max_tokens,
            "output_schema": None,
        }
        if t.repeat_idx is not None:
            payload["repeat_idx"] = t.repeat_idx
        return payload

    key_a = stable_key(make_payload(task_a))
    key_b = stable_key(make_payload(task_b))
    assert key_a != key_b


# ---------------------------------------------------------------------------
# End-to-end integration (mocked client)
# ---------------------------------------------------------------------------


async def test_end_to_end_mock(
    sample_questions_df: pd.DataFrame,
    mock_bet_client: LLMClient,
    tmp_path: Path,
) -> None:
    """3 questions x 2 repeats -> 6 JSONL rows, each with 16 bet + 2 kb cells."""
    m_repeats = 2
    expanded = expand_with_repeats(sample_questions_df, m_repeats=m_repeats)
    factories = build_truthfulqa_task_factories(STAKES_VARIANTS)

    out = tmp_path / "bet_out.jsonl"
    await run_multi_task_inference(
        df=expanded,
        client=mock_bet_client,
        task_factories=factories,
        out_json=out,
        n=len(expanded),
        mode=RunMode.OVERWRITE,
        extra_fields=extra_fields_fn,
        combine_outputs=make_combine_outputs(STAKES_VARIANTS),
    )

    latest = load_latest_records(out)
    assert len(latest) == 6

    for rid, record in latest.items():
        assert record["status"] == "ok"
        bvo = BetVariantsOutput.model_validate(record["output"])
        assert len(bvo.cells) == 16


async def test_end_to_end_row_keys(
    sample_questions_df: pd.DataFrame,
    mock_bet_client: LLMClient,
    tmp_path: Path,
) -> None:
    """Every output row must have required keys."""
    expanded = expand_with_repeats(sample_questions_df, m_repeats=1)
    factories = build_truthfulqa_task_factories(STAKES_VARIANTS)

    out = tmp_path / "bet_keys.jsonl"
    await run_multi_task_inference(
        df=expanded,
        client=mock_bet_client,
        task_factories=factories,
        out_json=out,
        n=len(expanded),
        mode=RunMode.OVERWRITE,
        extra_fields=extra_fields_fn,
        combine_outputs=make_combine_outputs(STAKES_VARIANTS),
    )

    latest = load_latest_records(out)
    assert len(latest) == 3
    for record in latest.values():
        required = {"question_id", "correct_answer", "incorrect_answer", "repeat_idx", "output", "status"}
        assert required.issubset(record.keys())


# ---------------------------------------------------------------------------
# SKIP_DONE / REDO_FAILED
# ---------------------------------------------------------------------------


async def test_skip_done_no_op(
    sample_questions_df: pd.DataFrame,
    mock_bet_client: LLMClient,
    tmp_path: Path,
) -> None:
    """Second run with SKIP_DONE is a no-op (no new LLM calls)."""
    expanded = expand_with_repeats(sample_questions_df, m_repeats=1)
    factories = build_truthfulqa_task_factories(STAKES_VARIANTS)
    out = tmp_path / "skip.jsonl"

    # First run
    await run_multi_task_inference(
        df=expanded, client=mock_bet_client, task_factories=factories,
        out_json=out, n=len(expanded), mode=RunMode.OVERWRITE,
        extra_fields=extra_fields_fn, combine_outputs=make_combine_outputs(STAKES_VARIANTS),
    )

    mock_bet_client.run.reset_mock()

    # Second run — SKIP_DONE
    await run_multi_task_inference(
        df=expanded, client=mock_bet_client, task_factories=factories,
        out_json=out, n=len(expanded), mode=RunMode.SKIP_DONE,
        extra_fields=extra_fields_fn, combine_outputs=make_combine_outputs(STAKES_VARIANTS),
    )

    assert mock_bet_client.run.call_count == 0


async def test_redo_failed_reruns_errors(
    sample_questions_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """REDO_FAILED re-runs only rows with status == 'error'."""
    expanded = expand_with_repeats(sample_questions_df, m_repeats=1)
    out = tmp_path / "redo.jsonl"

    # Seed the output with one error row and two ok rows
    records = {}
    for i, (rid, row) in enumerate(expanded.iterrows()):
        rec = {
            "prompt_row_id": int(rid),
            "model": "test-bet",
            "base_url": None,
            "status": "error" if i == 0 else "ok",
        }
        records[int(rid)] = rec

    from inference.batch_inference import atomic_rewrite_jsonl
    atomic_rewrite_jsonl(out, records)

    # Build a client that succeeds
    client = LLMClient(
        endpoint=EndpointConfig(provider="openai_compat", model="test-bet", api_key="x"),
        cache_dir=None,
    )

    async def fake_run(task: InferenceTask) -> Dict[str, Any]:
        return {
            "response_content": "Friend",
            "raw_response": {},
        }

    client.run = AsyncMock(side_effect=fake_run)

    factories = build_truthfulqa_task_factories(STAKES_VARIANTS)
    await run_multi_task_inference(
        df=expanded, client=client, task_factories=factories,
        out_json=out, n=len(expanded), mode=RunMode.REDO_FAILED,
        extra_fields=extra_fields_fn, combine_outputs=make_combine_outputs(STAKES_VARIANTS),
    )

    # Only the one error row should have been re-run (18 tasks per row)
    assert client.run.call_count == 18

    latest = load_latest_records(out)
    for record in latest.values():
        assert record["status"] == "ok"
