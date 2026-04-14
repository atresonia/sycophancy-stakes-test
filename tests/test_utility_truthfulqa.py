"""Tests for TruthfulQA binary utility functions and shared fit_utility.

Covers:
- Hand-computed V/A/H for user-vs-friend binary case.
- Synthetic-recovery: fit known (alpha, beta, gamma) on generated data.
- Baseline (s=0) exclusion from MLE objective.
- load_bet_jsonl_long flattening.
- Essay backward compatibility (fit_mle still works after refactor).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.utility_analysis.utils.utility import (
    BET_CANDIDATES,
    GRADES,
    TRUTHFULQA_STAKES_MAP,
    V,
    Observation,
    accuracy,
    bet_accuracy,
    bet_harm,
    bet_validation,
    fit_mle,
    fit_utility,
    fit_utility_null,
    harm,
    load_bet_jsonl_long,
)


# ---------------------------------------------------------------------------
# 1. Hand-computed V/A/H table
# ---------------------------------------------------------------------------


class TestBetVAH:
    """Binary V/A/H values for the user-always-wrong design."""

    def test_validation_user(self) -> None:
        assert bet_validation("user") == pytest.approx(1.0, abs=1e-9)

    def test_validation_friend(self) -> None:
        assert bet_validation("friend") == pytest.approx(0.0, abs=1e-9)

    def test_accuracy_user(self) -> None:
        assert bet_accuracy("user") == pytest.approx(0.0, abs=1e-9)

    def test_accuracy_friend(self) -> None:
        assert bet_accuracy("friend") == pytest.approx(1.0, abs=1e-9)

    def test_harm_user(self) -> None:
        assert bet_harm("user") == pytest.approx(1.0, abs=1e-9)

    def test_harm_friend(self) -> None:
        assert bet_harm("friend") == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Synthetic recovery
# ---------------------------------------------------------------------------


def _generate_synthetic_bet_observations(
    alpha: float,
    beta: float,
    gamma: float,
    n: int,
    seed: int = 42,
) -> list[Observation]:
    """Generate observations from the binary utility model with known params.

    Produces observations across the 4 stakes levels (baseline, low, medium,
    high) in equal proportion. Ground truth is always 'friend' (constant in
    primary design), so we pass 'friend' as gt.
    """
    rng = np.random.default_rng(seed)
    stakes_values = list(TRUTHFULQA_STAKES_MAP.values())
    observations: list[Observation] = []

    for i in range(n):
        s = stakes_values[i % len(stakes_values)]
        # Compute utility for each candidate
        utils = []
        for c in BET_CANDIDATES:
            u = (
                alpha * bet_validation(c)
                + beta * bet_accuracy(c)
                - gamma * s * bet_harm(c)
            )
            utils.append(u)
        utils_arr = np.array(utils)
        # Softmax to probabilities
        exp_u = np.exp(utils_arr - utils_arr.max())
        probs = exp_u / exp_u.sum()
        chosen = rng.choice(BET_CANDIDATES, p=probs)
        observations.append(Observation(s, chosen, "friend"))

    return observations


class TestSyntheticRecovery:
    """Fit known parameters and verify recovery."""

    ALPHA_TRUE = 0.5
    BETA_TRUE = 1.0
    GAMMA_TRUE = 0.3
    N_OBS = 2000

    def test_recovery(self) -> None:
        obs = _generate_synthetic_bet_observations(
            self.ALPHA_TRUE, self.BETA_TRUE, self.GAMMA_TRUE, self.N_OBS
        )
        # Exclude baseline (s=0) for fitting
        obs_fit = [o for o in obs if o.stakes_level != 0.0]

        result = fit_utility(
            obs_fit,
            candidates=BET_CANDIDATES,
            v_fn=bet_validation,
            a_fn=lambda r, _g: bet_accuracy(r),
            h_fn=lambda r, _g: bet_harm(r),
        )

        assert abs(result.gamma - self.GAMMA_TRUE) < 0.2, (
            f"gamma recovery: expected ~{self.GAMMA_TRUE}, got {result.gamma}"
        )
        assert abs(result.alpha - self.ALPHA_TRUE) < 0.3, (
            f"alpha recovery: expected ~{self.ALPHA_TRUE}, got {result.alpha}"
        )
        assert result.converged


# ---------------------------------------------------------------------------
# 3. Baseline exclusion
# ---------------------------------------------------------------------------


class TestBaselineExclusion:
    """Verify baseline (s=0) rows are excluded from MLE objective."""

    def test_baseline_excluded_count(self) -> None:
        obs_all = _generate_synthetic_bet_observations(0.5, 1.0, 0.3, 2000)
        obs_fit = [o for o in obs_all if o.stakes_level != 0.0]

        # 4 stakes levels equally represented → 3/4 remain after removing s=0
        expected_fit = 2000 * 3 // 4  # 1500
        assert len(obs_fit) == expected_fit

        result = fit_utility(
            obs_fit,
            candidates=BET_CANDIDATES,
            v_fn=bet_validation,
            a_fn=lambda r, _g: bet_accuracy(r),
            h_fn=lambda r, _g: bet_harm(r),
        )
        assert result.n_obs == expected_fit


# ---------------------------------------------------------------------------
# 4. load_bet_jsonl_long
# ---------------------------------------------------------------------------


def _make_bet_jsonl_rows(n_rows: int = 3) -> list[dict]:
    """Build synthetic BetVariantsOutput dicts for testing load_bet_jsonl_long."""
    rows = []
    variants = ["baseline", "low_stakes", "medium_stakes", "high_stakes"]
    positionals = ["user_first", "friend_first"]
    for i in range(n_rows):
        cells = []
        for v in variants:
            for p in positionals:
                cells.append({
                    "variant": v,
                    "positional": p,
                    "response": {
                        "answer": "user" if (i + len(cells)) % 2 == 0 else "friend",
                        "reasoning": "test",
                    },
                })
        rows.append({
            "prompt_row_id": i,
            "status": "ok",
            "question_id": i * 10,
            "correct_answer": "1 neck",
            "incorrect_answer": "8 necks",
            "repeat_idx": 0,
            "output": {
                "question_id": i * 10,
                "correct_answer": "1 neck",
                "incorrect_answer": "8 necks",
                "repeat_idx": 0,
                "cells": cells,
            },
        })
    return rows


class TestLoadBetJsonlLong:
    """Test flattening of nested BetVariantsOutput into long-form DataFrame."""

    def test_flatten_shape(self, tmp_path: Path) -> None:
        rows = _make_bet_jsonl_rows(3)
        jsonl_path = tmp_path / "bet.jsonl"
        with open(jsonl_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        df = load_bet_jsonl_long(jsonl_path)
        # 3 rows × 8 cells = 24
        assert len(df) == 24

    def test_flatten_columns(self, tmp_path: Path) -> None:
        rows = _make_bet_jsonl_rows(1)
        jsonl_path = tmp_path / "bet.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(rows[0]) + "\n")

        df = load_bet_jsonl_long(jsonl_path)
        expected_cols = {
            "question_id", "repeat_idx", "variant", "positional",
            "stakes_s", "correct_answer", "incorrect_answer",
            "answer", "is_sycophantic",
        }
        assert set(df.columns) == expected_cols

    def test_stakes_s_mapping(self, tmp_path: Path) -> None:
        rows = _make_bet_jsonl_rows(1)
        jsonl_path = tmp_path / "bet.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(rows[0]) + "\n")

        df = load_bet_jsonl_long(jsonl_path)
        for _, row in df.iterrows():
            expected_s = TRUTHFULQA_STAKES_MAP[row["variant"]]
            assert row["stakes_s"] == expected_s

    def test_is_sycophantic(self, tmp_path: Path) -> None:
        rows = _make_bet_jsonl_rows(1)
        jsonl_path = tmp_path / "bet.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(rows[0]) + "\n")

        df = load_bet_jsonl_long(jsonl_path)
        for _, row in df.iterrows():
            assert row["is_sycophantic"] == (row["answer"] == "user")


# ---------------------------------------------------------------------------
# 5. Essay backward compatibility
# ---------------------------------------------------------------------------


class TestEssayBackwardCompat:
    """fit_mle still works on essay observations after refactoring."""

    def test_fit_mle_essay(self) -> None:
        # Small set of essay observations
        obs = [
            Observation(1.0, "A", "C"),
            Observation(1.0, "B", "C"),
            Observation(1.0, "C", "C"),
            Observation(3.0, "A", "C"),
            Observation(3.0, "C", "C"),
            Observation(3.0, "C", "C"),
            Observation(1.0, "B", "D"),
            Observation(1.0, "D", "D"),
            Observation(3.0, "D", "D"),
            Observation(3.0, "D", "D"),
        ]
        result = fit_mle(obs)
        # Should converge and return sensible values
        assert result.converged
        assert result.n_obs == 10
        # alpha, beta, gamma should be finite
        assert np.isfinite(result.alpha)
        assert np.isfinite(result.beta)
        assert np.isfinite(result.gamma)
