"""Generate TruthfulQA bet-stakes responses across stakes x positional x repeats.

For each question, runs 8 LLM calls (4 stakes variants x 2 positionals) per
repeat. Output ``.jsonl`` rows conform to ``BetVariantsOutput`` -- one row per
(question_id, repeat_idx), each containing 8 nested cells.

Usage:
    python scripts/utility_analysis/generate_llm_resp_truthfulqa_bet.py \\
        --num_ex 50 --m_repeats 20 \\
        --model gemini-2.5-flash --provider gemini \\
        --in_csv data/truthfulqa/TruthfulQA.csv \\
        --out_json data/truthfulqa/llm_resp_truthfulqa_bet_gemini.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

from data_loading.truthfulqa import load_truthfulqa
from inference.prompts.truthfulqa import (
    POSITIONALS,
    STAKES_VARIANTS,
    create_bet_task,
)
from inference.schemas.truthfulqa import BetAnswerOutput, BetCellResult, BetVariantsOutput
from inference.task import InferenceTask
from scripts.utility_analysis.experiment_runner import build_parser, run_experiment


def expand_with_repeats(df: pd.DataFrame, m_repeats: int) -> pd.DataFrame:
    """Cross-join questions with repeat indices.

    Args:
        df: DataFrame with question rows (must have ``question_id`` column).
        m_repeats: Number of repeats per question.

    Returns:
        Expanded DataFrame with ``repeat_idx`` column and unique integer index
        computed as ``question_id * m_repeats + repeat_idx``.
    """
    repeats = pd.DataFrame({"repeat_idx": range(m_repeats)})
    expanded = df.merge(repeats, how="cross")
    expanded.index = expanded["question_id"] * m_repeats + expanded["repeat_idx"]
    expanded.index.name = None
    return expanded


def load_data(args: Any) -> pd.DataFrame:
    """Load TruthfulQA data and expand with repeats.

    Args:
        args: Parsed CLI namespace with ``in_csv``, ``num_ex``, ``m_repeats``.

    Returns:
        Expanded DataFrame ready for batch inference.
    """
    df = load_truthfulqa(Path(args.in_csv), num_ex=args.num_ex, seed=1234)
    return expand_with_repeats(df, args.m_repeats)


def build_truthfulqa_task_factories(
    variants: Dict[str, str],
) -> Dict[str, Callable[[pd.Series], InferenceTask]]:
    """Build 8 task factories keyed as ``{variant}_{positional}``.

    Each factory sets ``repeat_idx`` on the task so that repeated runs of the
    same prompt produce distinct cache entries.

    Args:
        variants: Stakes variant dict (e.g. ``STAKES_VARIANTS``).

    Returns:
        Dict of 8 task factories.
    """
    factories: Dict[str, Callable[[pd.Series], InferenceTask]] = {}
    for variant_name, stakes_block in variants.items():
        for positional in POSITIONALS:
            key = f"{variant_name}_{positional}"

            def factory(
                row: pd.Series,
                _block: str = stakes_block,
                _pos: str = positional,
            ) -> InferenceTask:
                task = create_bet_task(
                    question=row["question"],
                    user_claim=row["incorrect_answer"],
                    friend_claim=row["correct_answer"],
                    stakes_block=_block,
                    positional=_pos,
                )
                task.repeat_idx = int(row["repeat_idx"])
                return task

            factories[key] = factory
    return factories


def extra_fields_fn(row: pd.Series) -> Dict[str, Any]:
    """Extract TruthfulQA-specific fields from a row for the output record.

    Args:
        row: DataFrame row with question metadata.

    Returns:
        Dict with question_id, correct_answer, incorrect_answer, repeat_idx.
    """
    return {
        "question_id": int(row["question_id"]),
        "correct_answer": row["correct_answer"],
        "incorrect_answer": row["incorrect_answer"],
        "repeat_idx": int(row["repeat_idx"]),
    }


def make_combine_outputs() -> Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
    """Create a callback that assembles 8 cell outputs into a ``BetVariantsOutput``.

    Returns:
        Callable that receives ``(outputs, base_record)`` and returns a dict
        with an ``"output"`` key containing the validated ``BetVariantsOutput``.
    """

    def combine(
        outputs: Dict[str, Any],
        base_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        cells: list[BetCellResult] = []
        for key, resp_dict in outputs.items():
            key_name = key.removesuffix("_output")
            for pos in POSITIONALS:
                if key_name.endswith(pos):
                    variant = key_name.removesuffix(f"_{pos}")
                    cells.append(
                        BetCellResult(
                            variant=variant,
                            positional=pos,
                            response=BetAnswerOutput(**resp_dict),
                        )
                    )
                    break

        bvo = BetVariantsOutput(
            question_id=base_record["question_id"],
            correct_answer=base_record["correct_answer"],
            incorrect_answer=base_record["incorrect_answer"],
            repeat_idx=base_record["repeat_idx"],
            cells=cells,
        )
        return {"output": bvo.model_dump()}

    return combine


if __name__ == "__main__":
    parser = build_parser("Generate TruthfulQA bet-stakes responses")
    parser.add_argument(
        "--m_repeats", type=int, default=50,
        help="Number of independent repeats per question (default 50).",
    )
    parser.set_defaults(in_csv="data/truthfulqa/TruthfulQA.csv")
    args = parser.parse_args()
    run_experiment(
        STAKES_VARIANTS,
        args,
        load_data=load_data,
        build_task_factories=build_truthfulqa_task_factories,
        extra_fields_fn=extra_fields_fn,
        combine_outputs=make_combine_outputs(),
    )
