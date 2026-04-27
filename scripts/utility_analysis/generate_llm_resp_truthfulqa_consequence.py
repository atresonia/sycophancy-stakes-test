"""Generate TruthfulQA consequence-stakes responses across stakes x positional x repeats.

Like the monetary bet experiment, but uses qualitative consequence-based stakes
(fun → dinner → rent) instead of dollar amounts ($10/$500/$5,000).  The premise
strings are length-matched at 17 words to control for prompt-length confounds.

For each question, runs 8 LLM calls (4 stakes variants x 2 positionals) per
repeat.  Output ``.jsonl`` rows conform to ``BetVariantsOutput`` -- one row per
(question_id, repeat_idx), each containing 8 nested cells.

Usage:
    python scripts/utility_analysis/generate_llm_resp_truthfulqa_consequence.py \\
        --num_ex 50 --m_repeats 20 \\
        --model gemini-2.5-flash --provider gemini \\
        --in_csv data/truthfulqa/TruthfulQA.csv \\
        --out_json data/truthfulqa/llm_resp_consequence_gemini.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

from data_loading.truthfulqa import load_truthfulqa
from inference.prompts.truthfulqa import (
    CONSEQUENCE_STAKES_VARIANTS,
    KNOWLEDGE_POSITIONS,
    POSITIONALS,
    USER_STANCE,
    create_bet_task,
    create_knowledge_baseline_task,
)
from inference.schemas.truthfulqa import (
    BetAnswerOutput,
    BetCellResult,
    BetVariantsOutput,
    KnowledgeBaselineCellResult,
    parse_mc_response,
    parse_one_word_response,
)
from inference.task import InferenceTask
from scripts.utility_analysis.experiment_runner import build_parser, run_experiment

# Reuse expand_with_repeats, load_data, extra_fields_fn from the monetary script
from scripts.utility_analysis.generate_llm_resp_truthfulqa_bet import (
    expand_with_repeats,
    extra_fields_fn,
    load_data,
)


def build_consequence_task_factories(
    variants: Dict[str, str],
    temperature: float = 1.0,
) -> Dict[str, Callable[[pd.Series], InferenceTask]]:
    """Build 18 task factories: 2 knowledge baseline + 16 consequence-bet.

    Knowledge baseline keys: ``knowledge_baseline_correct_{A|B}``
    Bet keys: ``{variant}_{user_stance}_{positional}``

    Each factory passes the full consequence premise via ``premise_override``
    to ``create_bet_task``, bypassing the monetary ``stakes_block`` interpolation.

    Args:
        variants: Consequence stakes variant dict (``CONSEQUENCE_STAKES_VARIANTS``).
        temperature: Sampling temperature for LLM calls.

    Returns:
        Dict of 18 task factories.
    """
    factories: Dict[str, Callable[[pd.Series], InferenceTask]] = {}

    # --- 2 knowledge baseline factories ---
    for pos in KNOWLEDGE_POSITIONS:
        key = f"knowledge_baseline_correct_{pos}"

        def kb_factory(
            row: pd.Series,
            _pos: str = pos,
        ) -> InferenceTask:
            task = create_knowledge_baseline_task(
                question=row["question"],
                correct_answer=row["correct_answer"],
                incorrect_answer=row["incorrect_answer"],
                correct_position=_pos,
                temperature=temperature,
            )
            task.repeat_idx = int(row["repeat_idx"])
            return task

        factories[key] = kb_factory

    # --- 16 consequence-bet factories ---
    for variant_name, premise_string in variants.items():
        for stance in USER_STANCE:
            for positional in POSITIONALS:
                key = f"{variant_name}_{stance}_{positional}"

                def factory(
                    row: pd.Series,
                    _premise: str = premise_string,
                    _stance: str = stance,
                    _pos: str = positional,
                ) -> InferenceTask:
                    task = create_bet_task(
                        question=row["question"],
                        correct_answer=row["correct_answer"],
                        incorrect_answer=row["incorrect_answer"],
                        stakes_block="",  # unused; premise_override takes precedence
                        positional=_pos,
                        user_stance=_stance,
                        premise_override=_premise,
                        temperature=temperature,
                    )
                    task.output_schema = None  # raw text; parsed via parse_one_word_response
                    task.repeat_idx = int(row["repeat_idx"])
                    return task

                factories[key] = factory
    return factories


def make_combine_outputs(
    variants: Dict[str, str],
) -> Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
    """Create a callback that assembles 18 cell outputs into a ``BetVariantsOutput``.

    Identical to the monetary version -- output schema is the same.

    Args:
        variants: Consequence stakes variant dict.

    Returns:
        Callable that receives ``(outputs, base_record)`` and returns a dict
        with an ``"output"`` key containing the validated ``BetVariantsOutput``.
    """
    # Pre-build key → (variant, stance, positional) lookup for bet keys
    bet_key_meta: Dict[str, tuple[str, str, str]] = {}
    for vname in variants:
        for stance in USER_STANCE:
            for pos in POSITIONALS:
                bet_key_meta[f"{vname}_{stance}_{pos}"] = (vname, stance, pos)

    def combine(
        outputs: Dict[str, Any],
        base_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        cells: list[BetCellResult] = []
        kb_cells: list[KnowledgeBaselineCellResult] = []

        for key, raw_response in outputs.items():
            key_name = key.removesuffix("_output")
            raw_text = raw_response if isinstance(raw_response, str) else str(raw_response)

            # --- Knowledge baseline keys ---
            if key_name.startswith("knowledge_baseline_correct_"):
                correct_position = key_name.removeprefix("knowledge_baseline_correct_")
                answer = parse_mc_response(raw_text)
                kb_cells.append(
                    KnowledgeBaselineCellResult(
                        correct_position=correct_position,
                        answer=answer,
                        raw_response=raw_text,
                        is_correct=answer == correct_position,
                    )
                )
                continue

            # --- Bet keys ---
            meta = bet_key_meta.get(key_name)
            if meta is None:
                raise ValueError(f"Unrecognized cell key: {key_name!r}")
            variant, stance, pos = meta
            parsed_answer = parse_one_word_response(raw_text)
            cells.append(
                BetCellResult(
                    variant=variant,
                    positional=pos,
                    user_stance=stance,
                    response=BetAnswerOutput(
                        answer=parsed_answer,
                        raw_response=raw_text,
                    ),
                )
            )

        bvo = BetVariantsOutput(
            question_id=base_record["question_id"],
            correct_answer=base_record["correct_answer"],
            incorrect_answer=base_record["incorrect_answer"],
            repeat_idx=base_record["repeat_idx"],
            cells=cells,
            knowledge_baseline_cells=kb_cells,
        )
        return {"output": bvo.model_dump()}

    return combine


if __name__ == "__main__":
    parser = build_parser("Generate TruthfulQA consequence-stakes responses")
    parser.add_argument(
        "--m_repeats", type=int, default=50,
        help="Number of independent repeats per question (default 50).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for LLM calls (default 1.0).",
    )
    parser.set_defaults(in_csv="data/truthfulqa/TruthfulQA.csv")
    args = parser.parse_args()
    run_experiment(
        CONSEQUENCE_STAKES_VARIANTS,
        args,
        load_data=load_data,
        build_task_factories=lambda v: build_consequence_task_factories(v, temperature=args.temperature),
        extra_fields_fn=extra_fields_fn,
        combine_outputs=make_combine_outputs(CONSEQUENCE_STAKES_VARIANTS),
    )
