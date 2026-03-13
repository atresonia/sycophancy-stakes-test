"""Tests for utils.helpers."""
from pathlib import Path

import pandas as pd
import pytest

from utils.helpers import load_df_from_jsonl


def test_load_df_from_jsonl_basic(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"a": 1, "b": 2}\n'
        '{"a": 3, "b": 4}\n'
    )
    df = load_df_from_jsonl(path)
    assert len(df) == 2
    assert list(df.columns) == ["a", "b"]
    assert df["a"].tolist() == [1, 3]


def test_load_df_from_jsonl_with_nested_flatten(tmp_path: Path) -> None:
    path = tmp_path / "nested.jsonl"
    path.write_text('{"x": 1, "nested": {"low": "L", "high": "H"}}\n')
    df = load_df_from_jsonl(path)
    assert "nested.low" in df.columns or "nested" in df.columns
    assert len(df) == 1


def test_load_df_from_jsonl_cols_filter(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1, "b": 2, "c": 3}\n')
    df = load_df_from_jsonl(path, cols=["a", "c"])
    assert list(df.columns) == ["a", "c"]


def test_load_df_from_jsonl_rename(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"old_name": 1}\n')
    df = load_df_from_jsonl(path, rename_cols={"old_name": "new_name"})
    assert "new_name" in df.columns
