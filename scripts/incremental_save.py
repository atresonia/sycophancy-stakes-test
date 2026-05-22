"""Resume-safe incremental writing for long-running async experiments.

Each row's result is appended to `out_path` the moment its `process_row`
coroutine completes. On a re-run, rows whose `id_col` value already appears
in the output CSV are skipped, so a crash mid-run loses at most the rows
still in flight.
"""
import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Set

import pandas as pd
from tqdm.asyncio import tqdm as tqdm_asyncio


ProcessRow = Callable[[pd.Series], Awaitable[Dict[str, Any]]]


def load_done_ids(out_path: Path | None, id_col: str) -> Set[Any]:
    """Return the set of IDs already written to `out_path`."""
    if out_path is None or not out_path.exists():
        return set()
    done = pd.read_csv(out_path, usecols=[id_col])
    return set(done[id_col].tolist())


async def run_with_resume(
    df: pd.DataFrame,
    out_path: Path | None,
    id_col: str,
    process_row: ProcessRow,
    desc: str = "rows",
    overwrite: bool = False,
    row_concurrency: int | None = None,
) -> pd.DataFrame:
    """Run `process_row` on each row of `df`, appending results to `out_path`.

    If `out_path` already exists, rows whose `id_col` value is present are
    skipped so the run picks up where it left off. Pass `overwrite=True` to
    delete the existing file and start fresh. Per-row writes are serialized
    through an asyncio lock; concurrency in `process_row` itself is unaffected.

    `row_concurrency` caps how many rows run concurrently. `None` (default)
    means all rows start at once. Set to 1 when each row already issues
    enough sub-requests to saturate the API — that way rows complete one
    after another and the progress bar actually advances.
    """
    if overwrite and out_path is not None and out_path.exists():
        print(f"Overwriting: removing {out_path}")
        out_path.unlink()

    done_ids = load_done_ids(out_path, id_col)
    if done_ids:
        df = df[~df[id_col].isin(done_ids)]
        print(f"Resuming: {len(done_ids)} rows already in {out_path}, {len(df)} to go")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_header: list[str] | None = None
    if out_path is not None and out_path.exists():
        existing_header = list(pd.read_csv(out_path, nrows=0).columns)

    write_lock = asyncio.Lock()
    header_needed = out_path is not None and not out_path.exists()
    schema_checked = False
    row_sem = asyncio.Semaphore(row_concurrency if row_concurrency else max(1, len(df)))

    async def run_and_save(row: pd.Series) -> Dict[str, Any]:
        async with row_sem:
            result = await process_row(row)
            if out_path is not None:
                nonlocal header_needed, schema_checked
                async with write_lock:
                    if not schema_checked and existing_header is not None:
                        new_cols = list(result.keys())
                        if new_cols != existing_header:
                            raise ValueError(
                                f"Schema mismatch with existing {out_path}.\n"
                                f"  existing columns: {existing_header}\n"
                                f"  new result keys:  {new_cols}\n"
                                f"Pass overwrite=True (or --overwrite) to start fresh."
                            )
                        schema_checked = True
                    pd.DataFrame([result]).to_csv(
                        out_path, mode="a", header=header_needed, index=False
                    )
                    header_needed = False
            return result

    results = await tqdm_asyncio.gather(
        *(run_and_save(row) for _, row in df.iterrows()),
        desc=desc,
        total=len(df),
    )
    if out_path is not None and out_path.exists():
        return pd.read_csv(out_path)
    return pd.DataFrame(results)

