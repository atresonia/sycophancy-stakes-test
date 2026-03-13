"""Tests for inference.batch_inference."""
from __future__ import annotations

import asyncio
from pathlib import Path

import orjson
import pandas as pd
import pytest

from inference.batch_inference import (
    RunMode,
    load_latest_records,
    atomic_rewrite_jsonl,
    run_batch_inference,
    run_multi_task_inference,
)
from inference.llm_inference import InferenceTask, LLMClient, EndpointConfig


def test_load_latest_records_empty_path(tmp_path: Path) -> None:
    out = tmp_path / "nonexistent.jsonl"
    assert load_latest_records(out) == {}


def test_load_latest_records_keeps_latest_per_id(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    with out.open("wb") as f:
        f.write(orjson.dumps({"prompt_row_id": 0, "status": "old"}) + b"\n")
        f.write(orjson.dumps({"prompt_row_id": 0, "status": "ok"}) + b"\n")
        f.write(orjson.dumps({"prompt_row_id": 1, "status": "ok"}) + b"\n")
    latest = load_latest_records(out)
    assert len(latest) == 2
    assert latest[0]["status"] == "ok"
    assert latest[1]["status"] == "ok"


def test_atomic_rewrite_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    records = {
        1: {"prompt_row_id": 1, "a": 1},
        0: {"prompt_row_id": 0, "a": 0},
    }
    atomic_rewrite_jsonl(out, records)
    lines = out.read_bytes().split(b"\n")
    lines = [l for l in lines if l]
    assert len(lines) == 2
    first = orjson.loads(lines[0])
    assert first["prompt_row_id"] == 0
    second = orjson.loads(lines[1])
    assert second["prompt_row_id"] == 1


@pytest.fixture
def mock_client() -> LLMClient:
    client = LLMClient(
        endpoint=EndpointConfig(provider="openai_compat", model="test", api_key="x"),
        cache_dir=None,
    )
    async def fake_run(task: InferenceTask):
        return {"response_content": {"out": str(task.get_user_prompt())[:20]}, "raw_response": {}}
    client.run = fake_run
    return client


@pytest.mark.asyncio
async def test_run_batch_inference_writes_jsonl(
    mock_client: LLMClient,
    sample_df_aita: pd.DataFrame,
    tmp_path: Path,
) -> None:
    out = tmp_path / "batch.jsonl"
    def task_factory(row: pd.Series) -> InferenceTask:
        return InferenceTask(user_prompt=row["prompt"])

    await run_batch_inference(
        df=sample_df_aita,
        client=mock_client,
        task_factory=task_factory,
        out_json=out,
        n=1,
        mode=RunMode.OVERWRITE,
    )
    assert out.exists()
    latest = load_latest_records(out)
    assert len(latest) == 1
    assert latest[0]["status"] == "ok"
    assert "output" in latest[0]


@pytest.mark.asyncio
async def test_run_multi_task_inference(
    mock_client: LLMClient,
    sample_df_stakes_outputs: pd.DataFrame,
    tmp_path: Path,
) -> None:
    out = tmp_path / "multi.jsonl"
    def low_factory(row: pd.Series) -> InferenceTask:
        return InferenceTask(user_prompt=row["low_variant_prompt"])
    def high_factory(row: pd.Series) -> InferenceTask:
        return InferenceTask(user_prompt=row["high_variant_prompt"])

    await run_multi_task_inference(
        df=sample_df_stakes_outputs,
        client=mock_client,
        task_factories={"low_var": low_factory, "high_var": high_factory},
        out_json=out,
        n=1,
        mode=RunMode.OVERWRITE,
    )
    latest = load_latest_records(out)
    assert len(latest) == 1
    assert latest[0]["status"] == "ok"
    assert "low_var_output" in latest[0]
    assert "high_var_output" in latest[0]
