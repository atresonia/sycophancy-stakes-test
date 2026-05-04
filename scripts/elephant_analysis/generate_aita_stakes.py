"""Generate AITA stakes-variant judgments (7 variants per prompt).

Runs 7 LLM calls per prompt (1 baseline + 3 low_stakes + 3 high_stakes) at
temperature 0.0. Output ``.jsonl`` rows conform to ``AITAStakesOutput``.

Usage:
    # OpenAI
    python scripts/elephant_analysis/generate_aita_stakes.py --num_ex 50 \\
        --model gpt-5.2 --provider openai_compat \\
        --out_json data/aita/llm_resp_aita_stakes_openai.jsonl
    # Gemini
    python scripts/elephant_analysis/generate_aita_stakes.py --num_ex 50 \\
        --model gemini-2.5-flash --provider gemini \\
        --out_json data/aita/llm_resp_aita_stakes_gemini.jsonl
    # Anthropic
    python scripts/elephant_analysis/generate_aita_stakes.py --num_ex 50 \\
        --model claude-sonnet-4-20250514 --provider anthropic \\
        --out_json data/aita/llm_resp_aita_stakes_claude.jsonl
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import pandas as pd

from inference.prompts.elephant import AITA_STAKES_VARIANTS, create_aita_judgment_task
from inference.schemas.elephant import (
    AITACellResult, AITAJudgment, AITAStakesOutput, parse_yta_nta_response,
)
from inference.task import InferenceTask
from scripts.utility_analysis.experiment_runner import build_parser, run_experiment


def load_data(args: Any) -> pd.DataFrame:
    df = pd.read_csv(args.in_csv, index_col=0)
    df.index.name = "prompt_id"
    if args.num_ex is not None and args.num_ex < len(df):
        df = df.sample(n=args.num_ex, random_state=1234)
    return df


def build_aita_task_factories(
    variants: Dict[str, str],
) -> Dict[str, Callable[[pd.Series], InferenceTask]]:
    return {
        name: (lambda row, _t=text: create_aita_judgment_task(
            prompt=row["prompt"], stakes_text=_t, temperature=0.0,
        ))
        for name, text in variants.items()
    }


def combine_outputs(
    outputs: Dict[str, Any], base_record: Dict[str, Any],
) -> Dict[str, Any]:
    cells = [
        AITACellResult(
            variant=key.removesuffix("_output"),
            judgment=AITAJudgment(
                answer=parse_yta_nta_response(str(raw)), raw_response=str(raw)),
        )
        for key, raw in outputs.items()
    ]
    return {"output": AITAStakesOutput(
        prompt_id=base_record["prompt_id"], cells=cells).model_dump()}


if __name__ == "__main__":
    parser = build_parser("Generate AITA stakes-variant judgments")
    parser.set_defaults(in_csv="data/aita/AITA-YTA.csv")
    args = parser.parse_args()
    run_experiment(
        AITA_STAKES_VARIANTS, args,
        load_data=load_data,
        build_task_factories=build_aita_task_factories,
        extra_fields_fn=lambda row: {"prompt_id": int(row.name), "ytanta": row["ytanta"]},
        combine_outputs=combine_outputs,
    )
