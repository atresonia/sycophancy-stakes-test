"""
Script to generate stakes variants for a given dataset.

Mainly meant to be used on AITA YTA dataset.

Usage:
- Overwrite: python scripts/generate_stakes_variants.py --num_ex 10 --model gpt-5.2 --in_csv AITA-YTA.csv --out_json aita-yta-stakes-gen-openai.jsonl
- Redo failed: python scripts/generate_stakes_variants.py --num_ex 10 --model gpt-5.2 --in_csv AITA-YTA.csv --out_json aita-yta-stakes-gen-openai.jsonl --mode redo_failed
- Skip done: python scripts/generate_stakes_variants.py --num_ex 10 --model gpt-5.2 --in_csv AITA-YTA.csv --out_json aita-yta-stakes-gen-openai.jsonl --mode skip_done
"""
import argparse
import asyncio
from inference.batch_inference import run_batch_inference, RunMode
from inference.llm_inference import EndpointConfig, InferenceTask, LLMClient, create_stakes_generation_task
import pandas as pd
from pathlib import Path
import os
import json

def make_stakes_generation_task(row: pd.Series) -> InferenceTask:
    return create_stakes_generation_task(
        prompt=row["prompt"],
        reference_label=row["ytanta"],
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ex", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--provider", type=str, choices=["openai_compat", "anthropic"], default="openai_compat")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--in_csv", type=str, default="data/AITA-YTA.csv")
    parser.add_argument("--out_json", type=str, default="data/aita-yta-stakes-gen-openai.jsonl")
    parser.add_argument("--mode", type=lambda s: RunMode[s], choices=list(RunMode), default=RunMode.OVERWRITE)
    parser.add_argument("--force_ids", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true", help="Disable LLM response caching")
    args = parser.parse_args()
    
    force_ids = [int(fid) for fid in args.force_ids.split(",")] if args.force_ids else None
    num_ex = args.num_ex

    # Set API key based on provider if not explicitly provided
    if args.api_key is None:
        if args.provider == "openai_compat":
            args.api_key = os.getenv("OPENAI_API_KEY")
        else:
            args.api_key = os.getenv("ANTHROPIC_API_KEY")

    print(f"API key: {args.api_key}")
    print(f"Generating stakes variants for {args.in_csv} with {args.num_ex} rows using {args.model} in {args.mode.name} mode...")

    df = pd.read_csv(args.in_csv)
    # sample num_ex rows
    df = df.sample(n=num_ex, random_state=1234)

    # generate stakes variants
    client = LLMClient(endpoint=EndpointConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        cache_dir=None if args.no_cache else ".llm_cache",
    ))
    asyncio.run(run_batch_inference(
        df=df,
        client=client,
        task_factory=make_stakes_generation_task,
        out_json=Path(args.out_json),
        n=num_ex,
        mode=args.mode,
        force_ids=force_ids,
    ))
    print(f"Wrote results to: {Path(args.out_json)}")

if __name__ == "__main__":
    main()