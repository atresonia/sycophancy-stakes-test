"""
Batch inference utilities for running tasks over dataframes.

Usage:
    from batch_inference import run_batch_inference, RunMode
    from llm_inference import LLMClient, EndpointConfig, InferenceTask
    
    # Define how to create a task from a row
    def make_task(row):
        return InferenceTask(
            user_prompt=row["prompt"],
            system_prompt="You are a judge."
        )
    
    # Run batch
    await run_batch_inference(
        df=my_df,
        client=client,
        task_factory=make_task,
        out_json=Path("results.jsonl"),
        n=100,
    )
"""

import traceback
import asyncio
import orjson
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List
from tqdm import tqdm
import pandas as pd

from inference.client import LLMClient
from inference.task import InferenceTask


class RunMode(Enum):
    """How to handle existing results."""
    SKIP_DONE = 1      # Skip rows that already have results
    REDO_FAILED = 2    # Redo rows that failed, skip successful ones
    OVERWRITE = 3      # Overwrite all results


def load_latest_records(out_json: Path) -> Dict[int, Dict[str, Any]]:
    """Load latest record per prompt_row_id from existing jsonl file."""
    latest: Dict[int, Dict[str, Any]] = {}
    if not out_json.exists():
        return latest

    with out_json.open("rb") as f:
        for line in f:
            try:
                obj = orjson.loads(line)
            except Exception:
                continue
            rid = obj.get("prompt_row_id")
            if isinstance(rid, int):
                latest[rid] = obj
    return latest


def atomic_rewrite_jsonl(out_json: Path, records: Dict[int, Dict[str, Any]]) -> None:
    """Rewrite jsonl with only the latest record per id."""
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    with tmp.open("wb") as f:
        for rid in sorted(records.keys()):
            f.write(orjson.dumps(records[rid]) + b"\n")
    tmp.replace(out_json)


def _is_ok(rec: Dict[str, Any]) -> bool:
    return rec.get("status") == "ok"


async def run_batch_inference(
    df: pd.DataFrame,
    client: LLMClient,
    task_factory: Callable[[pd.Series], InferenceTask],
    out_json: Optional[Path] = None,
    n: int = 10,
    mode: RunMode = RunMode.SKIP_DONE,
    force_ids: Optional[List[int]] = None,
    extra_fields: Optional[Callable[[pd.Series], Dict[str, Any]]] = None,
    done_ids: Optional[List[int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Run batch inference over a dataframe.

    Args:
        df: DataFrame with rows to process (index is used as row id)
        client: LLMClient instance
        task_factory: Function that takes a row and returns an InferenceTask
        out_json: Optional path to persist results as a jsonl. When provided,
                  prior results are loaded to support SKIP_DONE / REDO_FAILED
                  across runs and the file is atomically rewritten after each
                  run. When omitted, results are only returned in-memory.
        n: Max number of rows to process
        mode: How to handle existing results
        force_ids: Explicitly rerun these row ids regardless of mode
        extra_fields: Optional function to extract extra fields from row for output
        done_ids: Row IDs already known to be successfully completed (e.g. from an
                  external CSV). Used for SKIP_DONE / REDO_FAILED when out_json is
                  not provided.
    """
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
    latest = load_latest_records(out_json) if out_json is not None else {}

    # External successes are kept as a separate set — never written to latest/jsonl
    # so the final atomic_rewrite only contains real inference results.
    external_done: set[int] = set(done_ids) if (done_ids and mode != RunMode.OVERWRITE) else set()

    # Determine which rows to process
    if mode == RunMode.OVERWRITE:
        latest = {}
        candidates = list(df.index)
    elif mode == RunMode.REDO_FAILED:
        # Skip only confirmed successes; retry failures and unseen rows.
        candidates = [rid for rid in df.index
                      if rid not in external_done and not (rid in latest and _is_ok(latest[rid]))]
    elif mode == RunMode.SKIP_DONE:
        # Skip anything already tried (success or failure) or externally done.
        candidates = [rid for rid in df.index if rid not in latest and rid not in external_done]
    else:
        raise ValueError(f"Invalid mode={mode}")

    if force_ids:
        forced = [rid for rid in force_ids if rid in df.index]
        candidates = forced + [rid for rid in candidates if rid not in force_ids]

    candidates = list(candidates)[:n]
    selected_df = df.loc[candidates]
    
    print(f"Selected {len(selected_df)} rows (mode={mode.name}).")

    async def process_one(row_id: int, row: pd.Series) -> Dict[str, Any]:
        task = task_factory(row)
        
        base_record = {
            "prompt_row_id": int(row_id),
            "model": client.endpoint.model,
            "base_url": client.endpoint.base_url,
        }
        
        if extra_fields:
            base_record.update(extra_fields(row))
        
        try:
            resp = await client.run(task)
            return {
                **base_record,
                "status": "ok",
                "output": resp["response_content"],
            }
        except Exception as e:
            return {
                **base_record,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    tasks = [
        asyncio.create_task(process_one(int(rid), row))
        for rid, row in selected_df.iterrows()
    ]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing", unit="row"):
        record = await fut
        latest[record["prompt_row_id"]] = record
        
    if out_json is not None:
        atomic_rewrite_jsonl(out_json, latest)
        print(f"Wrote results to: {out_json}")
    return latest


async def run_multi_task_inference(
    df: pd.DataFrame,
    client: LLMClient,
    task_factories: Dict[str, Callable[[pd.Series], InferenceTask]],
    out_json: Path,
    n: int = 10,
    mode: RunMode = RunMode.SKIP_DONE,
    force_ids: Optional[List[int]] = None,
    extra_fields: Optional[Callable[[pd.Series], Dict[str, Any]]] = None,
    combine_outputs: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> None:
    """
    Run multiple inference tasks per row (e.g., low and high variants).
    
    Args:
        task_factories: Dict mapping output key names to task factory functions
                       e.g., {"low_var": make_low_task, "high_var": make_high_task}
        combine_outputs: Optional. Given the dict of {key_output: value}, return extra
                         fields to merge into the record (e.g. {"output": combined_dict}).
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    latest = load_latest_records(out_json)
    
    if mode == RunMode.OVERWRITE:
        latest = {}
        candidates = list(df.index)
    elif mode == RunMode.REDO_FAILED:
        candidates = [rid for rid in df.index if (rid not in latest) or (not _is_ok(latest[rid]))]
    elif mode == RunMode.SKIP_DONE:
        candidates = [rid for rid in df.index if rid not in latest]

    if force_ids:
        forced = [rid for rid in force_ids if rid in df.index]
        candidates = forced + [rid for rid in candidates if rid not in force_ids]

    candidates = list(candidates)[:n]
    selected_df = df.loc[candidates]
    
    print(f"Selected {len(selected_df)} rows (mode={mode.name}).")

    async def process_one(row_id: int, row: pd.Series) -> Dict[str, Any]:
        base_record = {
            "prompt_row_id": int(row_id),
            "model": client.endpoint.model,
            "base_url": client.endpoint.base_url,
        }
        
        if extra_fields:
            base_record.update(extra_fields(row))
        
        try:
            keys = list(task_factories.keys())
            resps = await asyncio.gather(*[client.run(factory(row)) for factory in task_factories.values()])
            outputs = {f"{key}_output": resp["response_content"] for key, resp in zip(keys, resps)}
            extra = combine_outputs(outputs, base_record) if combine_outputs else {}
            return {
                **base_record,
                "status": "ok",
                **outputs,
                **extra,
            }
        except Exception as e:
            return {
                **base_record,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    tasks = [
        asyncio.create_task(process_one(int(rid), row))
        for rid, row in selected_df.iterrows()
    ]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing", unit="row"):
        record = await fut
        latest[record["prompt_row_id"]] = record
        
    atomic_rewrite_jsonl(out_json, latest)
    print(f"Wrote results to: {out_json}")
