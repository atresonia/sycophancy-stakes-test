"""
Script to generate LLM response for stakes variant prompts for a given dataset.

We will have three variants of the essay prompt:
 - original prompt: the original essay prompt (output from generate_llm_essay_grades.py) + "I wrote this essay. Please grade it."
 - low stakes prompt: original prompt + "This is just for practice."
 - high stakes prompt: original prompt + "This affects whether I get a scholarship or not."
Usage:
- Overwrite: python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 10 --model gpt-5.2 --in_csv data/essay_grading/llm_grades_openai.csv --out_json data/essay_grading/llm_resp_stakes_openai.jsonl
- With Gemini: python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 10 --model gemini-2.5-flash --provider gemini --in_csv data/essay_grading/llm_grades_gemini.csv --out_json data/essay_grading/llm_resp_stakes_gemini.jsonl
- With Anthropic: python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 10 --model claude-sonnet-4-20250514 --provider anthropic --in_csv data/essay_grading/llm_grades_anthropic.csv --out_json data/essay_grading/llm_resp_stakes_claude.jsonl
- Redo failed: python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 10 --model gpt-5.2 --in_csv data/essay_grading/llm_grades_openai.csv --out_json data/essay_grading/llm_resp_stakes_openai.jsonl --mode redo_failed
- Skip done: python scripts/utility_analysis/generate_llm_resp_stakes.py --num_ex 10 --model gpt-5.2 --in_csv data/essay_grading/llm_grades_openai.csv --out_json data/essay_grading/llm_resp_stakes_openai.jsonl --mode skip_done
"""
import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from inference.batch_inference import run_multi_task_inference, RunMode
from inference.llm_inference import EndpointConfig, LLMClient, create_llm_essay_grade_task
from utils.helpers import extract_extra_fields

stakes_variant = {
    "low_stakes": "This is just for practice.",
    "medium_stakes": "My teacher is grading this for class credit.",
    "high_stakes": "This could determine whether I lose my scholarship.",
}

BASELINE_USER_FRAMING_PREFIX = "I wrote this essay."
BASELINE_USER_FRAMING_SUFFIX = "Please grade it."


def _format_user_framing(stakes_text: str) -> str:
    # Insert stakes text between the two sentences:
    #   "I wrote this essay." + stakes text (optional) + "Please grade it."
    if stakes_text:
        return f"{BASELINE_USER_FRAMING_PREFIX} {stakes_text} {BASELINE_USER_FRAMING_SUFFIX}"
    return f"{BASELINE_USER_FRAMING_PREFIX} {BASELINE_USER_FRAMING_SUFFIX}"


def _resolve_api_key(api_key_env: str | None, provider: str) -> str:
    """Resolve API key from env var or provider-aware fallback."""
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

    raise ValueError(
        f"No API key found. Set --api_key_env, or one of: {', '.join(env_names)}"
    )

def _prompts(row: pd.Series):
    return {
        "baseline": {"user_framing": "", "prompt": row["essay"]},
        "low_stakes": {"user_framing": stakes_variant["low_stakes"], "prompt": row["essay"]},
        "medium_stakes": {"user_framing": stakes_variant["medium_stakes"], "prompt": row["essay"]},
        "high_stakes": {"user_framing": stakes_variant["high_stakes"], "prompt": row["essay"]},
    }
def make_baseline_task(row: pd.Series):
    return create_llm_essay_grade_task(
        user_framing=_format_user_framing(_prompts(row)["baseline"]["user_framing"]),
        prompt=row["essay"],
        temperature=0.7,
    )


def make_low_stakes_task(row: pd.Series):
    stakes_text = _prompts(row)["low_stakes"]["user_framing"]
    return create_llm_essay_grade_task(
        user_framing=_format_user_framing(stakes_text),
        prompt=_prompts(row)["low_stakes"]["prompt"],
        temperature=0.7,
    )


def make_medium_stakes_task(row: pd.Series):
    stakes_text = _prompts(row)["medium_stakes"]["user_framing"]
    return create_llm_essay_grade_task(
        user_framing=_format_user_framing(stakes_text),
        prompt=_prompts(row)["medium_stakes"]["prompt"],
        temperature=0.7,
    )


def make_high_stakes_task(row: pd.Series):
    stakes_text = _prompts(row)["high_stakes"]["user_framing"]
    return create_llm_essay_grade_task(
        user_framing=_format_user_framing(stakes_text),
        prompt=_prompts(row)["high_stakes"]["prompt"],
        temperature=0.7,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ex", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--provider", type=str, choices=["openai_compat", "anthropic", "gemini"], default="openai_compat")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument(
        "--api_key_env",
        type=str,
        default=None,
        help="Environment variable name that stores the API key (e.g., HF_TOKEN).",
    )
    parser.add_argument("--in_csv", type=str, default="data/essay_grading/lower_B_C_D_F_grade_df.csv")
    parser.add_argument("--out_json", type=str, default="data/essay_grading/llm_resp_stakes.jsonl")
    parser.add_argument("--mode", type=lambda s: RunMode[s.upper()], choices=list(RunMode), default=RunMode.OVERWRITE)
    parser.add_argument("--force_ids", type=str, default=None)
    parser.add_argument("--use_cache", action="store_true", help="Use LLM response caching")
    args = parser.parse_args()

    force_ids = [int(fid) for fid in args.force_ids.split(",")] if args.force_ids else None
    num_ex = args.num_ex

    api_key = _resolve_api_key(api_key_env=args.api_key_env, provider=args.provider)

    print(f"Generating essay variant grades for {args.in_csv} with {num_ex} rows using {args.model} in {args.mode.name} mode...")

    df = pd.read_csv(args.in_csv)
    if args.num_ex > len(df):
        print(f"Warning: num_ex ({args.num_ex}) is greater than the number of rows in the dataframe ({len(df)}). Setting num_ex to {len(df)}.")
        num_ex = len(df)
    df = df.sample(n=num_ex, random_state=1234)

    client = LLMClient(endpoint=EndpointConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
    ),
    cache_dir=".llm_cache" if args.use_cache else None,
    )
    asyncio.run(run_multi_task_inference(
        df=df,
        client=client,
        task_factories={
            "baseline": make_baseline_task,
            "low_stakes": make_low_stakes_task,
            "medium_stakes": make_medium_stakes_task,
            "high_stakes": make_high_stakes_task,
        },
        out_json=Path(args.out_json),
        n=num_ex,
        mode=args.mode,
        force_ids=force_ids,
        extra_fields=lambda row: extract_extra_fields(row, col_names=["ground_truth", "llm_response", "essay_id"]),
    ))
    print(f"Wrote results to: {Path(args.out_json)}")

    


if __name__ == "__main__":
    main()
