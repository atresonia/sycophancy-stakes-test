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
import os
import sys
from pathlib import Path

import pandas as pd

from inference.batch_inference import run_batch_inference, RunMode
from inference.client import EndpointConfig, LLMClient
from inference.task import InferenceTask
from inference.prompts.essay import create_llm_essay_grade_task


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

    # Read existing CSV once — used to skip already-done rows (done_ids) and to
    # fill them back after the join (the batch runner only processes new rows, so
    # previously-successful grades would otherwise be NaN in the output).
    done_ids: list[int] | None = None
    existing_grades: pd.Series | None = None
    if args.mode in (RunMode.REDO_FAILED, RunMode.SKIP_DONE) and args.out_csv:
        try:
            existing_df = pd.read_csv(args.out_csv).set_index("essay_id")
            valid = existing_df["llm_response"].notna() & (existing_df["llm_response"].str.strip() != "")
            done_ids = existing_df.index[valid].astype(int).tolist()
            existing_grades = existing_df.loc[valid, "llm_response"]
            print(f"Found {len(done_ids)} already-completed rows in {args.out_csv}.", file=sys.stderr)
        except FileNotFoundError:
            pass

    results = asyncio.run(run_batch_inference(
        df=df,
        client=client,
        task_factory=make_llm_essay_grade_task,
        n=num_ex,
        mode=args.mode,
        force_ids=force_ids,
        done_ids=done_ids,
    ))

    error_rows = []
    ok_rows: dict[int, dict] = {}
    for rid, r in results.items():
        if r.get("status") == "ok":
            ok_rows[rid] = r
        else:
            error_rows.append(r)

    if error_rows:
        print(f"WARNING: {len(error_rows)}/{len(results)} rows failed:")
        for r in error_rows:
            print(f"  Row {r['prompt_row_id']}: {r.get('error', 'unknown error')}\n{r.get('traceback', '')}")

    if not ok_rows and existing_grades is None:
        raise RuntimeError("All inference calls failed — check errors above.")

    def extract_grade(output) -> str:
        if isinstance(output, dict):
            return output.get("grade", "")
        return str(output).strip()

    results_df = (
        pd.DataFrame([
            {"prompt_row_id": rid, "llm_response": extract_grade(r["output"])}
            for rid, r in ok_rows.items()
        ]).set_index("prompt_row_id")
        if ok_rows else pd.DataFrame(columns=["llm_response"])
    )

    out_df = df[["essay", "true_grade"]].join(results_df).rename(columns={"true_grade": "ground_truth"}).reset_index(names="essay_id")

    # Fill in grades for rows that were skipped (already successful in a prior run).
    if existing_grades is not None:
        out_df = out_df.set_index("essay_id")
        out_df["llm_response"] = out_df["llm_response"].fillna(existing_grades)
        out_df = out_df.reset_index()

    if args.out_csv is None:
        print(out_df.to_csv(index=False), end="")
    else:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {len(out_df)} rows to: {out_path}")


if __name__ == "__main__":
    main()
