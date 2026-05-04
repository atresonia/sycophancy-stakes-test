"""
Script to generate sycophancy evaluation for a given dataset.

Mainly meant to be used on aita-yta-stakes-outputs-openai.jsonl dataset. 
Can be generated using the generate_stakes_outputs.py script.

Usage:
- Overwrite: python scripts/generate_syc_eval.py --in_json aita-yta-stakes-outputs-openai.jsonl --out_json aita-yta-syc-eval-openai.jsonl
- Redo failed: python scripts/generate_syc_eval.py --in_json aita-yta-stakes-outputs-openai.jsonl --out_json aita-yta-syc-eval-openai.jsonl --mode redo_failed
- Skip done: python scripts/generate_syc_eval.py --in_json aita-yta-stakes-outputs-openai.jsonl --out_json aita-yta-syc-eval-openai.jsonl --mode skip_done
- With Gemini (structured output): python scripts/generate_syc_eval.py --in_json aita-yta-stakes-outputs-openai.jsonl --out_json aita-yta-syc-eval-gemini.jsonl --model gemini-2.0-flash --provider gemini
- With Anthropic: python scripts/generate_syc_eval.py --in_json aita-yta-stakes-outputs-openai.jsonl --out_json aita-yta-syc-eval-claude.jsonl --model claude-sonnet-4-20250514 --provider anthropic
"""
import argparse
import asyncio
import os
from pathlib import Path
import pandas as pd
from inference.batch_inference import run_batch_inference, RunMode
from inference.client import EndpointConfig, LLMClient, Provider
from inference.task import InferenceTask
from inference.prompts.aita import create_sycophancy_eval_task
from utils.helpers import load_df_from_jsonl

def make_sycophancy_eval_task(row: pd.Series) -> InferenceTask:
    return create_sycophancy_eval_task(
        low_stakes_prompt=row["low_var_prompt"],
        low_stakes_response=row["low_var_output"],
        high_stakes_prompt=row["high_var_prompt"],
        high_stakes_response=row["high_var_output"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ex", type=int, default=None)
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--provider", type=str, choices=["openai_compat", "anthropic", "gemini"], default="openai_compat")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--in_json", type=str, default="data/aita-yta-variants-gen-openai.jsonl")
    parser.add_argument("--out_json", type=str, default="data/aita-yta-syc-eval-openai.jsonl")
    parser.add_argument("--mode", type=lambda s: RunMode[s], choices=list(RunMode), default=RunMode.OVERWRITE)
    parser.add_argument("--force_ids", type=str, default=None)
    parser.add_argument("--use_cache", action="store_true", help="Use LLM response caching")
    args = parser.parse_args()
    force_ids = [int(fid) for fid in args.force_ids.split(",")] if args.force_ids else None
    num_ex = args.num_ex

    print(f"Generating sycophancy evaluation for {args.in_json} with {args.num_ex} rows using {args.model} in {args.mode.name} mode...")

    df = load_df_from_jsonl(Path(args.in_json))
    # Use prompt_row_id as index so eval output matches stakes_output / stakes_variants for joining
    if "prompt_row_id" in df.columns:
        df = df.set_index("prompt_row_id")
    # load all if num_ex is None
    if num_ex is None:
        num_ex = len(df)
    df = df.sample(n=num_ex, random_state=1234)

    print(df)

    # Set API key based on provider if not explicitly provided
    api_key = args.api_key
    if api_key is None:
        if args.provider == "openai_compat":
            api_key = os.getenv("OPENAI_API_KEY")
        elif args.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            api_key = os.getenv("GEMINI_API_KEY")

    client = LLMClient(endpoint=EndpointConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
    ),
    cache_dir=".llm_cache" if args.use_cache else None,
    )
    asyncio.run(run_batch_inference(
        df=df,
        client=client,
        task_factory=make_sycophancy_eval_task,
        out_json=Path(args.out_json),
        n=num_ex,
        mode=args.mode,
        force_ids=force_ids,
    ))
    print(f"Wrote results to: {Path(args.out_json)}")

if __name__ == "__main__":
    main()