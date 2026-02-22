"""
Script to generate stakes outputs (YTA/NTA judgments) for low and high variants.

Takes the stakes-variants dataset (with low_variant_prompt and high_variant_prompt) (ex: data/aita-yta-stakes-gen-claude.jsonl)
and runs inference to get model judgments for each variant.

Usage:
- Overwrite: python scripts/generate_stakes_outputs.py --in_json data/aita-yta-stakes-gen-claude.jsonl --out_json data/aita-yta-variants-gen-openai.jsonl
- Redo failed: python scripts/generate_stakes_outputs.py --in_json data/aita-yta-stakes-gen-claude.jsonl --out_json data/aita-yta-variants-gen-openai.jsonl --mode REDO_FAILED
- Skip done: python scripts/generate_stakes_outputs.py --in_json data/aita-yta-stakes-gen-claude.jsonl --out_json data/aita-yta-variants-gen-openai.jsonl --mode SKIP_DONE
"""
import argparse
import asyncio
import os
from pathlib import Path
from inference.batch_inference import run_multi_task_inference, RunMode
from inference.llm_inference import EndpointConfig, InferenceTask, LLMClient, STAKES_OUTPUT_SYSTEM_PROMPT
import pandas as pd
from utils.helpers import load_df_from_jsonl


def make_low_variant_task(row: pd.Series) -> InferenceTask:
    """Create inference task for the low-stakes variant."""
    prompt = row["low_variant_prompt"]
    if not prompt or pd.isna(prompt):
        # Skip empty prompts by returning a dummy task that will be handled
        return InferenceTask(
            user_prompt="[EMPTY]",
            system_prompt=STAKES_OUTPUT_SYSTEM_PROMPT,
            max_tokens=10,
        )
    return InferenceTask(
        user_prompt=prompt + "\nOutput only YTA or NTA with a brief 1-2 sentence explanation.",
        system_prompt=STAKES_OUTPUT_SYSTEM_PROMPT,
        max_tokens=200,
    )


def make_high_variant_task(row: pd.Series) -> InferenceTask:
    """Create inference task for the high-stakes variant."""
    prompt = row["high_variant_prompt"]
    if not prompt or pd.isna(prompt):
        return InferenceTask(
            user_prompt="[EMPTY]",
            system_prompt=STAKES_OUTPUT_SYSTEM_PROMPT,
            max_tokens=10,
        )
    return InferenceTask(
        user_prompt=prompt + "\nOutput only YTA or NTA with a brief 1-2 sentence explanation.",
        system_prompt=STAKES_OUTPUT_SYSTEM_PROMPT,
        max_tokens=200,
    )


def extract_extra_fields(row: pd.Series) -> dict:
    """Extract the variant prompts to include in output for eval script."""
    return {
        "low_var_prompt": row.get("low_variant_prompt", ""),
        "high_var_prompt": row.get("high_variant_prompt", ""),
    }


def load_stakes_variants(jsonl_path: Path) -> pd.DataFrame:
    """Load stakes variants from jsonl, extracting nested variant prompts."""
    # Load and flatten JSON
    raw_df = pd.read_json(jsonl_path, lines=True)
    flat = pd.json_normalize(raw_df.to_dict("records"), sep=".")
    
    # Map the nested column names to our standard names
    col_mapping = {
        "output.variants.low.prompt": "low_variant_prompt",
        "output.variants.high.prompt": "high_variant_prompt",
    }
    
    # Apply renaming for columns that exist
    for old_col, new_col in col_mapping.items():
        if old_col in flat.columns:
            flat[new_col] = flat[old_col]
    
    # Keep only needed columns
    keep_cols = ["prompt_row_id", "low_variant_prompt", "high_variant_prompt"]
    df = flat[[c for c in keep_cols if c in flat.columns]].copy()
    
    print(f"Loaded columns: {list(df.columns)}")
    print(df)
    
    # Set prompt_row_id as index for proper tracking
    if "prompt_row_id" in df.columns:
        df = df.set_index("prompt_row_id")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate YTA/NTA judgments for stakes variants")
    parser.add_argument("--num_ex", type=int, default=None, help="Max number of rows to process (default: all)")
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--provider", type=str, choices=["openai_compat", "anthropic"], default="openai_compat")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--in_json", type=str, default="data/aita-yta-stakes-gen-claude.jsonl")
    parser.add_argument("--out_json", type=str, default="data/aita-yta-variants-gen-openai.jsonl")
    parser.add_argument("--mode", type=lambda s: RunMode[s], choices=list(RunMode), default=RunMode.OVERWRITE)
    parser.add_argument("--force_ids", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true", help="Disable LLM response caching")
    args = parser.parse_args()

    force_ids = [int(fid) for fid in args.force_ids.split(",")] if args.force_ids else None

    # Set API key based on provider if not explicitly provided
    if args.api_key is None:
        if args.provider == "openai_compat":
            args.api_key = os.getenv("OPENAI_API_KEY")
        else:
            args.api_key = os.getenv("ANTHROPIC_API_KEY")

    print(f"Generating stakes outputs for {args.in_json} using {args.model} in {args.mode.name} mode...")

    # Load the stakes variants dataset
    df = load_stakes_variants(Path(args.in_json))
    print(f"Loaded {len(df)} rows from {args.in_json}")

    # Limit number of examples if specified
    n = args.num_ex if args.num_ex else len(df)

    # Create client (disable cache if --no_cache flag is set)
    client = LLMClient(
        endpoint=EndpointConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        ),
        cache_dir=None if args.no_cache else ".llm_cache",
    )

    # Run multi-task inference (low and high variants per row)
    asyncio.run(run_multi_task_inference(
        df=df,
        client=client,
        task_factories={
            "low_var": make_low_variant_task,
            "high_var": make_high_variant_task,
        },
        out_json=Path(args.out_json),
        n=n,
        mode=args.mode,
        force_ids=force_ids,
        extra_fields=extract_extra_fields,
    ))

    print(f"Done! Results written to: {args.out_json}")


if __name__ == "__main__":
    main()
