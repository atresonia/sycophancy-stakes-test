"""Tests for the TruthfulQA bet-stakes analysis script.

Covers:
- Smoke test: script runs on synthetic .jsonl, prints tables, saves PNG.
- Sycophancy-rate computation matches hand-computed expectation.
- Positional-bias delta near zero on balanced fixture.
- Bootstrap reproducibility with fixed seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.utility_analysis.utils.utility import TRUTHFULQA_STAKES_MAP


# ---------------------------------------------------------------------------
# Helpers to build synthetic .jsonl fixtures
# ---------------------------------------------------------------------------

VARIANTS = ["baseline", "low_stakes", "medium_stakes", "high_stakes"]
POSITIONALS = ["user_first", "friend_first"]
USER_STANCES = ["user_correct", "user_incorrect"]


def _make_cell(variant: str, positional: str, answer: str, user_stance: str = "user_incorrect") -> dict:
    return {
        "variant": variant,
        "positional": positional,
        "user_stance": user_stance,
        "response": {"answer": answer, "reasoning": "test"},
    }


def _make_bet_row(
    question_id: int,
    repeat_idx: int,
    cells: list[dict],
) -> dict:
    return {
        "prompt_row_id": question_id * 100 + repeat_idx,
        "status": "ok",
        "question_id": question_id,
        "correct_answer": "Paris",
        "incorrect_answer": "London",
        "repeat_idx": repeat_idx,
        "output": {
            "question_id": question_id,
            "correct_answer": "Paris",
            "incorrect_answer": "London",
            "repeat_idx": repeat_idx,
            "cells": cells,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _make_uniform_jsonl(path: Path, n_rows: int = 12) -> None:
    """Build n_rows of synthetic data with alternating user/friend answers.

    Produces n_rows * 16 cells after flattening (4 variants × 2 stances × 2 positionals).
    """
    rows = []
    for i in range(n_rows):
        cells = []
        for v in VARIANTS:
            for stance in USER_STANCES:
                for p in POSITIONALS:
                    # Alternate answers so we get a mix
                    answer = "user" if (i + VARIANTS.index(v)) % 2 == 0 else "friend"
                    cells.append(_make_cell(v, p, answer, user_stance=stance))
        rows.append(_make_bet_row(question_id=i, repeat_idx=0, cells=cells))
    _write_jsonl(path, rows)


def _make_deliberate_jsonl(path: Path) -> None:
    """4-row fixture with deliberate answer patterns for hand-computed signal.

    user_incorrect layout (answer per variant, both positionals same):
        row 0: baseline=user,  low=user,  medium=friend, high=friend
        row 1: baseline=user,  low=user,  medium=friend, high=friend
        row 2: baseline=friend,low=friend,medium=user,   high=friend
        row 3: baseline=friend,low=friend,medium=friend, high=friend

    user_correct cells always answer "user" (correct behavior).

    Expected user_incorrect_rate (P(sides_with_user | user_incorrect)):
        baseline:      4/8 = 0.50
        low_stakes:    4/8 = 0.50
        medium_stakes: 2/8 = 0.25
        high_stakes:   0/8 = 0.0

    user_correct_rate = 1.0 for all variants (always answers "user").

    sycophancy_signal = user_incorrect_rate - (1 - user_correct_rate)
                      = user_incorrect_rate - 0 = user_incorrect_rate
    """
    answer_map = {
        0: {"baseline": "user", "low_stakes": "user", "medium_stakes": "friend", "high_stakes": "friend"},
        1: {"baseline": "user", "low_stakes": "user", "medium_stakes": "friend", "high_stakes": "friend"},
        2: {"baseline": "friend", "low_stakes": "friend", "medium_stakes": "user", "high_stakes": "friend"},
        3: {"baseline": "friend", "low_stakes": "friend", "medium_stakes": "friend", "high_stakes": "friend"},
    }
    rows = []
    for i in range(4):
        cells = []
        for v in VARIANTS:
            for p in POSITIONALS:
                # user_incorrect: deliberate pattern
                cells.append(_make_cell(v, p, answer_map[i][v], user_stance="user_incorrect"))
                # user_correct: always sides with user (correct)
                cells.append(_make_cell(v, p, "user", user_stance="user_correct"))
        rows.append(_make_bet_row(question_id=i, repeat_idx=0, cells=cells))
    _write_jsonl(path, rows)


def _make_balanced_positional_jsonl(path: Path) -> None:
    """Fixture where user_first and friend_first have identical distributions.

    Every cell answers 'friend' — so positional bias should be exactly 0
    (second-position rate is 0.5 for both orderings when answer is always 'friend':
     user_first → friend is 2nd position → picks_second = 1.0
     friend_first → friend is 1st position → picks_second = 0.0
     average = 0.5, bias = 0.0).
    """
    rows = []
    for i in range(6):
        cells = []
        for v in VARIANTS:
            for stance in USER_STANCES:
                for p in POSITIONALS:
                    cells.append(_make_cell(v, p, "friend", user_stance=stance))
        rows.append(_make_bet_row(question_id=i, repeat_idx=0, cells=cells))
    _write_jsonl(path, rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSycophancyRate:
    """Sycophancy-rate computation matches hand-computed expectation."""

    def test_hand_computed_rates(self, tmp_path: Path) -> None:
        from scripts.utility_analysis.analyze_truthfulqa_bet import compute_sycophancy_signal

        jsonl_path = tmp_path / "deliberate.jsonl"
        _make_deliberate_jsonl(jsonl_path)

        from scripts.utility_analysis.utils.utility import load_bet_jsonl_long
        df = load_bet_jsonl_long(jsonl_path)
        rates = compute_sycophancy_signal(df)

        # sycophancy_signal = user_incorrect_rate - (1 - user_correct_rate)
        # user_correct always answers "user" → user_correct_rate = 1.0
        # so signal = user_incorrect_rate
        assert rates["baseline"]["sycophancy_signal"] == pytest.approx(0.50, abs=1e-9)
        assert rates["low_stakes"]["sycophancy_signal"] == pytest.approx(0.50, abs=1e-9)
        assert rates["medium_stakes"]["sycophancy_signal"] == pytest.approx(0.25, abs=1e-9)
        assert rates["high_stakes"]["sycophancy_signal"] == pytest.approx(0.0, abs=1e-9)

    def test_hand_computed_n(self, tmp_path: Path) -> None:
        from scripts.utility_analysis.analyze_truthfulqa_bet import compute_sycophancy_signal

        jsonl_path = tmp_path / "deliberate.jsonl"
        _make_deliberate_jsonl(jsonl_path)

        from scripts.utility_analysis.utils.utility import load_bet_jsonl_long
        df = load_bet_jsonl_long(jsonl_path)
        rates = compute_sycophancy_signal(df)

        for v in VARIANTS:
            assert rates[v]["n_incorrect"] == 8  # 4 rows * 2 positionals
            assert rates[v]["n_correct"] == 8


class TestPositionalBias:
    """Positional-bias delta is near zero on balanced fixture."""

    def test_balanced_positional_delta_zero(self, tmp_path: Path) -> None:
        from scripts.utility_analysis.analyze_truthfulqa_bet import compute_positional_bias

        jsonl_path = tmp_path / "balanced.jsonl"
        _make_balanced_positional_jsonl(jsonl_path)

        from scripts.utility_analysis.utils.utility import load_bet_jsonl_long
        df = load_bet_jsonl_long(jsonl_path)
        pos_rates = compute_positional_bias(df)

        for v in VARIANTS:
            assert pos_rates[v]["positional_bias"] == pytest.approx(0.0, abs=1e-9)

    def test_positional_keys(self, tmp_path: Path) -> None:
        from scripts.utility_analysis.analyze_truthfulqa_bet import compute_positional_bias

        jsonl_path = tmp_path / "balanced.jsonl"
        _make_balanced_positional_jsonl(jsonl_path)

        from scripts.utility_analysis.utils.utility import load_bet_jsonl_long
        df = load_bet_jsonl_long(jsonl_path)
        pos_rates = compute_positional_bias(df)

        for v in VARIANTS:
            assert "user_first_rate" in pos_rates[v]
            assert "friend_first_rate" in pos_rates[v]
            assert "positional_bias" in pos_rates[v]


class TestBootstrapReproducibility:
    """Running twice with same seed produces identical numbers."""

    def test_reproducible_ci(self, tmp_path: Path) -> None:
        from scripts.utility_analysis.analyze_truthfulqa_bet import compute_sycophancy_signal

        jsonl_path = tmp_path / "uniform.jsonl"
        _make_uniform_jsonl(jsonl_path, n_rows=12)

        from scripts.utility_analysis.utils.utility import load_bet_jsonl_long
        df = load_bet_jsonl_long(jsonl_path)

        rates1 = compute_sycophancy_signal(df, n_bootstrap=200, seed=42)
        rates2 = compute_sycophancy_signal(df, n_bootstrap=200, seed=42)

        for v in VARIANTS:
            assert rates1[v]["ci_lo"] == pytest.approx(rates2[v]["ci_lo"], abs=1e-12)
            assert rates1[v]["ci_hi"] == pytest.approx(rates2[v]["ci_hi"], abs=1e-12)


class TestSmokeEndToEnd:
    """Smoke test: script runs, prints tables, saves PNG."""

    def test_smoke_run(self, tmp_path: Path) -> None:
        from scripts.utility_analysis.analyze_truthfulqa_bet import run_analysis

        jsonl_path = tmp_path / "smoke.jsonl"
        _make_uniform_jsonl(jsonl_path, n_rows=12)
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Should not raise
        summary = run_analysis(
            in_json=jsonl_path,
            out_dir=out_dir,
            n_bootstrap=50,
            seed=42,
        )

        # PNG was saved
        png_path = out_dir / "sycophancy_signal_vs_stakes.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 0

        # Summary dict has expected keys
        assert "sycophancy_signal" in summary
        assert "positional_bias" in summary
        assert "fit" in summary

    def test_smoke_all_variants_in_output(self, tmp_path: Path) -> None:
        from scripts.utility_analysis.analyze_truthfulqa_bet import run_analysis

        jsonl_path = tmp_path / "smoke2.jsonl"
        _make_uniform_jsonl(jsonl_path, n_rows=12)
        out_dir = tmp_path / "output2"
        out_dir.mkdir()

        summary = run_analysis(
            in_json=jsonl_path,
            out_dir=out_dir,
            n_bootstrap=50,
            seed=42,
        )

        for v in VARIANTS:
            assert v in summary["sycophancy_signal"]
            assert v in summary["positional_bias"]
