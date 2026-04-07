"""
Script to generate LLM essay grades for a given dataset.

Can be used on any dataset, but this is mainly meant to be used on the ASAP dataset (https://www.kaggle.com/competitions/asap-aes).
This script will generate an LLM essay grade for each essay in the dataset. We will measure the accuracy compared to the ground truth grades.
This output will be used to measure the sycophancy of the LLM on stakes variant prompts (filter "D" and "F" grades).
Usage:
- Overwrite: python scripts/utility_analysis/generate_llm_essay_grades.py --num_ex 10 --model gpt-5.2 --in_csv data/essay_grading/lower_D_F_grade_df.csv --out_csv data/essay_grading/llm_grades_openai.csv
- With Gemini: python scripts/utility_analysis/generate_llm_essay_grades.py --num_ex 10 --model gemini-2.5-flash --provider gemini --in_csv data/essay_grading/lower_D_F_grade_df.csv --out_csv data/essay_grading/llm_grades_gemini.csv
- With Anthropic: python scripts/utility_analysis/generate_llm_essay_grades.py --num_ex 10 --model claude-sonnet-4-20250514 --provider anthropic --in_csv data/essay_grading/lower_D_F_grade_df.csv --out_csv data/essay_grading/llm_grades_claude.csv
- Redo failed: python scripts/utility_analysis/generate_llm_essay_grades.py --num_ex 10 --model gpt-5.2 --in_csv data/essay_grading/lower_D_F_grade_df.csv --out_csv data/essay_grading/llm_grades_openai.csv --mode redo_failed
- Skip done: python scripts/utility_analysis/generate_llm_essay_grades.py --num_ex 10 --model gpt-5.2 --in_csv data/essay_grading/lower_D_F_grade_df.csv --out_csv data/essay_grading/llm_grades_openai.csv --mode skip_done
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from inference.batch_inference import run_batch_inference, RunMode
from inference.llm_inference import EndpointConfig, InferenceTask, LLMClient, create_llm_essay_grade_task


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
    parser.add_argument("--out_csv", type=str, default=None, help="Path to save output CSV. If omitted, prints to stdout.")
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

    print(f"Generating essay grades for {args.in_csv} with {num_ex} rows using {args.model} in {args.mode.name} mode...", file=sys.stderr)

    df = pd.read_csv(args.in_csv)
    if args.num_ex > len(df):
        print(f"Warning: num_ex ({args.num_ex}) is greater than the number of rows in the dataframe ({len(df)}). Setting num_ex to {len(df)}.")
        num_ex = len(df)
    df = df.sample(n=num_ex, random_state=1234).set_index("essay_id")

    client = LLMClient(
        endpoint=EndpointConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        ),
        cache_dir=".llm_cache" if args.use_cache else None,
    )

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_jsonl = Path(tmp.name)

    try:
        asyncio.run(run_batch_inference(
            df=df,
            client=client,
            task_factory=make_llm_essay_grade_task,
            out_json=tmp_jsonl,
            n=num_ex,
            mode=args.mode,
            force_ids=force_ids,
        ))

        # Load results and join with original df
        records = []
        with open(tmp_jsonl) as f:
            for line in f:
                records.append(json.loads(line))

        results_df = pd.DataFrame(records)

        # Report errors to help diagnose issues
        error_rows = results_df[results_df["status"] == "error"]
        if not error_rows.empty:
            print(f"WARNING: {len(error_rows)}/{len(results_df)} rows failed:")
            for _, row in error_rows.iterrows():
                print(f"  Row {row['prompt_row_id']}: {row.get('error', 'unknown error')} \n {row.get('traceback', 'no traceback')} ")

        results_df = results_df[results_df["status"] == "ok"].copy()
        if results_df.empty:
            raise RuntimeError("All inference calls failed — check errors above.")

        # Extract grade from structured output
        def extract_grade(output):
            if isinstance(output, dict):
                return output.get("grade", "")
            return str(output).strip()

        results_df["llm_response"] = results_df["output"].apply(extract_grade)
        results_df = results_df.set_index("prompt_row_id")[["llm_response"]]

        # index is essay_id in both df and results_df (prompt_row_id == essay_id)
        out_df = df[["essay", "true_grade"]].join(results_df).rename(columns={"true_grade": "ground_truth"}).reset_index(names="essay_id")

        if args.out_csv is None:
            print(out_df.to_csv(index=False), end="")
        else:
            out_path = Path(args.out_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_path, index=False)
            print(f"Wrote {len(out_df)} rows to: {out_path}")

    finally:
        tmp_jsonl.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
