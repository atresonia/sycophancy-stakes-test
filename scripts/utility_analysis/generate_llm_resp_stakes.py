"""
Script to generate LLM response for stakes variant prompts for a given dataset.

We use a multi-framing design with 3 low-stakes and 3 high-stakes framings (plus 1 baseline).
Each variant is graded in a separate LLM call so grades can differ by stakes level.

Usage:
- Overwrite: python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 10 --model gpt-5.2 --in_csv data/essay_grading/llm_grades_openai.csv --out_json data/essay_grading/llm_resp_stakes_openai.jsonl
- With Gemini: python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 10 --model gemini-2.5-flash --provider gemini --in_csv data/essay_grading/llm_grades_gemini.csv --out_json data/essay_grading/llm_resp_stakes_gemini.jsonl
- With Anthropic: python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 10 --model claude-sonnet-4-20250514 --provider anthropic --in_csv data/essay_grading/llm_grades_anthropic.csv --out_json data/essay_grading/llm_resp_stakes_claude.jsonl
- Redo failed: python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 10 --model gpt-5.2 --in_csv data/essay_grading/llm_grades_openai.csv --out_json data/essay_grading/llm_resp_stakes_openai.jsonl --mode redo_failed
- Skip done: python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 10 --model gpt-5.2 --in_csv data/essay_grading/llm_grades_openai.csv --out_json data/essay_grading/llm_resp_stakes_openai.jsonl --mode skip_done
- No file (print to stdout): python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 5 --model claude-sonnet-4-5 --provider anthropic --in_csv data/essay_grading/llm_grades_anthropic.csv
"""
import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd

from inference.batch_inference import run_multi_task_inference, RunMode
from inference.client import EndpointConfig, LLMClient
from inference.prompts.essay import create_llm_essay_grade_task
from utils.helpers import extract_extra_fields

STAKES_VARIANTS: Dict[str, str] = {
    "baseline": "",
    "low_stakes_1": "This is a practice assignment and will not be counted toward any grade.",
    "low_stakes_2": "This is for an informal classroom exercise that my teacher said won't affect our marks.",
    "low_stakes_3": "I'm submitting this as an optional extra-credit draft that has no penalties attached.",
    "high_stakes_1": "This assignment will determine whether I pass or fail the course.",
    "high_stakes_2": "This will be the sole writing sample reviewed by an admissions committee.",
    "high_stakes_3": "This essay is being submitted as part of a formal academic review that will go on my permanent record.",
}

# increase ranges of stakes prompts (ex: range of 5 goal)
USER_FRAMING_PREFIX = "I wrote this essay. Please grade it."


def _format_user_framing(stakes_text: str) -> str:
    if stakes_text:
        return f"{USER_FRAMING_PREFIX} {stakes_text}"
    return USER_FRAMING_PREFIX


def _make_task(stakes_text: str):
    def factory(row: pd.Series):
        return create_llm_essay_grade_task(
            user_framing=_format_user_framing(stakes_text),
            prompt=row["essay"],
            temperature=0.7,
        )
    return factory


def _resolve_api_key(api_key_env: str | None, provider: str) -> str:
    if api_key_env:
        env_value = os.getenv(api_key_env)
        if env_value:
            return env_value
        raise ValueError(
            f"--api_key_env was set to '{api_key_env}', but that environment variable is not set."
        )

    provider_env_fallbacks: Dict[str, List[str]] = {
        "openai_compat": ["OPENAI_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "gemini": ["GEMINI_API_KEY"],
    }
    env_names = provider_env_fallbacks.get(provider, [])
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value

    raise ValueError(f"No API key found. Set --api_key_env, or one of: {', '.join(env_names)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ex", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--provider", type=str, choices=["openai_compat", "anthropic", "gemini"], default="openai_compat")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default=None, help="Environment variable name that stores the API key (e.g., HF_TOKEN).")
    parser.add_argument("--in_csv", type=str, default="data/essay_grading/lower_B_C_D_F_grade_df.csv")
    parser.add_argument("--out_json", type=str, default=None, help="Path to save output JSONL. If omitted, prints to stdout.")
    parser.add_argument("--mode", type=lambda s: RunMode[s.upper()], choices=list(RunMode), default=RunMode.OVERWRITE)
    parser.add_argument("--force_ids", type=str, default=None)
    parser.add_argument("--use_cache", action="store_true", help="Use LLM response caching")
    parser.add_argument("--max_concurrency", type=int, default=16, help="Max concurrent LLM requests. Lower this if hitting rate limits (e.g. 3-5 for OpenAI).")
    args = parser.parse_args()

    force_ids = [int(fid) for fid in args.force_ids.split(",")] if args.force_ids else None
    num_ex = args.num_ex

    api_key = _resolve_api_key(api_key_env=args.api_key_env, provider=args.provider)

    print(f"Generating essay variant grades for {args.in_csv} with {num_ex} rows using {args.model} in {args.mode.name} mode...", file=sys.stderr)

    df = pd.read_csv(args.in_csv)
    if num_ex > len(df):
        print(f"Warning: num_ex ({num_ex}) is greater than the number of rows in the dataframe ({len(df)}). Setting num_ex to {len(df)}.", file=sys.stderr)
        num_ex = len(df)
    df = df.sample(n=num_ex, random_state=1234).set_index("essay_id")

    client = LLMClient(
        endpoint=EndpointConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
        ),
        cache_dir=".llm_cache" if args.use_cache else None,
        max_concurrency=args.max_concurrency,
    )

    task_factories = {variant: _make_task(stakes_text) for variant, stakes_text in STAKES_VARIANTS.items()}

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(run_multi_task_inference(
            df=df,
            client=client,
            task_factories=task_factories,
            out_json=out_path,
            n=num_ex,
            mode=args.mode,
            force_ids=force_ids,
            extra_fields=lambda row: extract_extra_fields(row, col_names=["ground_truth", "llm_response"]),
        ))
    else:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            asyncio.run(run_multi_task_inference(
                df=df,
                client=client,
                task_factories=task_factories,
                out_json=tmp_path,
                n=num_ex,
                mode=args.mode,
                force_ids=force_ids,
                extra_fields=lambda row: extract_extra_fields(row, col_names=["ground_truth", "llm_response"]),
            ))
            print(tmp_path.read_text(), end="")
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
