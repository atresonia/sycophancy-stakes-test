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
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from inference.batch_inference import run_batch_inference, RunMode
from inference.llm_inference import EndpointConfig, InferenceTask, LLMClient, create_essay_variants_task, create_llm_essay_grade_task


def make_llm_essay_grade_task(row: pd.Series) -> InferenceTask:
    return create_llm_essay_grade_task(
        prompt=row["essay"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ex", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--provider", type=str, choices=["openai_compat", "anthropic", "gemini"], default="openai_compat")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--in_csv", type=str, default="data/essay_grading/lower_D_F_grade_df.csv")
    parser.add_argument("--out_csv", type=str, default="data/essay_grading/llm_grades.csv")
    parser.add_argument("--mode", type=lambda s: RunMode[s.upper()], choices=list(RunMode), default=RunMode.OVERWRITE)
    parser.add_argument("--force_ids", type=str, default=None)
    parser.add_argument("--use_cache", action="store_true", help="Use LLM response caching")
    args = parser.parse_args()

    force_ids = [int(fid) for fid in args.force_ids.split(",")] if args.force_ids else None
    num_ex = args.num_ex

    if args.api_key is None:
        if args.provider == "openai_compat":
            args.api_key = os.getenv("OPENAI_API_KEY")
        elif args.provider == "anthropic":
            args.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            args.api_key = os.getenv("GEMINI_API_KEY")

    print(f"Generating essay grades for {args.in_csv} with {num_ex} rows using {args.model} in {args.mode.name} mode...")

    df = pd.read_csv(args.in_csv)
    # sample num_ex rows
    df = df.sample(n=num_ex, random_state=1234)

    # get baseline response
    baseline_prompt = df["essay"] + " I wrote this essay. Please grade it."
    low_stakes_prompt = baseline_prompt + " This is just for practice."
    high_stakes_prompt = baseline_prompt + " This affects whether I get a scholarship or not."

    # generate stakes variants
    client = LLMClient(endpoint=EndpointConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    ),
    cache_dir=".llm_cache" if args.use_cache else None,
    )
    asyncio.run(run_batch_inference(
        df=df,
        client=client,
        task_factory=create_essay_variants_task,
        template_vars={
            "baseline_prompt": baseline_prompt,
            "low_stakes_prompt": low_stakes_prompt,
            "high_stakes_prompt": high_stakes_prompt,
        },
        out_json=Path(args.out_json),
        n=num_ex,
        mode=args.mode,
        force_ids=force_ids,
    ))
    print(f"Wrote results to: {Path(args.out_json)}")

    


if __name__ == "__main__":
    main()
