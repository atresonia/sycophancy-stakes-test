"""Re-run only knowledge_baseline cells for existing TruthfulQA bet-stakes results.

Loads an existing JSONL, identifies rows where KB cells have empty raw_response,
re-runs the 2 KB tasks per row, patches the results in-place, and rewrites the file.

Usage:
    python scripts/utility_analysis/patch_kb_cells.py \
        --in_json data/truthfulqa/llm_resp_bet_gemini_v2.jsonl \
        --in_csv data/truthfulqa/TruthfulQA.csv \
        --model gemini-2.5-flash --provider gemini
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from data_loading.truthfulqa import load_truthfulqa
from inference.client import EndpointConfig, LLMClient
from inference.prompts.truthfulqa import (
    KNOWLEDGE_POSITIONS,
    create_knowledge_baseline_task,
)
from inference.schemas.truthfulqa import KnowledgeBaselineCellResult, parse_mc_response
from scripts.utility_analysis.experiment_runner import resolve_api_key


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("rb") as f:
        for line in f:
            records.append(orjson.loads(line))
    return records


def needs_patch(record: dict) -> bool:
    """Return True if any KB cell has an empty raw_response."""
    if record.get("status") != "ok":
        return False
    kb_cells = record.get("output", {}).get("knowledge_baseline_cells", [])
    if not kb_cells:
        return True
    return any(c["raw_response"] == "" for c in kb_cells)


async def rerun_kb_cells(
    client: LLMClient,
    question: str,
    correct_answer: str,
    incorrect_answer: str,
    repeat_idx: int,
) -> list[dict]:
    """Run the 2 knowledge baseline tasks and return cell dicts."""
    tasks = []
    for pos in KNOWLEDGE_POSITIONS:
        task = create_knowledge_baseline_task(
            question=question,
            correct_answer=correct_answer,
            incorrect_answer=incorrect_answer,
            correct_position=pos,
        )
        task.repeat_idx = repeat_idx
        tasks.append((pos, task))

    cells = []
    for pos, task in tasks:
        resp = await client.run(task)
        raw_text = resp["response_content"] if isinstance(resp["response_content"], str) else str(resp["response_content"])
        answer = parse_mc_response(raw_text)
        cells.append(
            KnowledgeBaselineCellResult(
                correct_position=pos,
                answer=answer,
                raw_response=raw_text,
                is_correct=answer == pos,
            ).model_dump()
        )
    return cells


async def patch_all(
    records: list[dict],
    client: LLMClient,
    questions_df: pd.DataFrame,
) -> int:
    """Patch KB cells in-place for all records that need it. Returns count patched."""
    # Build question_id -> row lookup
    q_lookup = {int(row["question_id"]): row for _, row in questions_df.iterrows()}

    to_patch = [(i, r) for i, r in enumerate(records) if needs_patch(r)]
    if not to_patch:
        print("No records need patching.", file=sys.stderr)
        return 0

    print(f"Patching {len(to_patch)} / {len(records)} records...", file=sys.stderr)

    async def do_one(idx: int, record: dict) -> tuple[int, list[dict]]:
        qid = record.get("question_id") or record.get("output", {}).get("question_id")
        repeat_idx = record.get("repeat_idx") or record.get("output", {}).get("repeat_idx", 0)
        q_row = q_lookup[qid]
        cells = await rerun_kb_cells(
            client,
            question=q_row["question"],
            correct_answer=q_row["correct_answer"],
            incorrect_answer=q_row["incorrect_answer"],
            repeat_idx=repeat_idx,
        )
        return idx, cells

    coros = [do_one(idx, rec) for idx, rec in to_patch]
    patched = 0
    for fut in tqdm(asyncio.as_completed(coros), total=len(coros), desc="Patching KB", unit="row"):
        idx, cells = await fut
        records[idx]["output"]["knowledge_baseline_cells"] = cells
        # Also update the top-level flattened fields
        for cell in cells:
            pos = cell["correct_position"]
            records[idx][f"knowledge_baseline_correct_{pos}_output"] = cell["raw_response"]
        patched += 1

    return patched


def main():
    parser = argparse.ArgumentParser(description="Patch empty knowledge_baseline cells in existing JSONL")
    parser.add_argument("--in_json", type=str, required=True)
    parser.add_argument("--in_csv", type=str, default="data/truthfulqa/TruthfulQA.csv")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--provider", type=str, choices=["openai_compat", "anthropic", "gemini"], required=True)
    parser.add_argument("--api_key_env", type=str, default=None)
    parser.add_argument("--max_concurrency", type=int, default=None)
    args = parser.parse_args()

    in_json = Path(args.in_json)
    records = load_records(in_json)
    print(f"Loaded {len(records)} records from {in_json}", file=sys.stderr)

    # Load all questions (no sampling; we only need the lookup)
    questions_df = load_truthfulqa(Path(args.in_csv))

    api_key = resolve_api_key(api_key_env=args.api_key_env, provider=args.provider)
    client = LLMClient(
        endpoint=EndpointConfig(
            provider=args.provider,
            model=args.model,
            api_key=api_key,
        ),
        cache_dir=None,
        max_concurrency=args.max_concurrency,
    )

    patched = asyncio.run(patch_all(records, client, questions_df))

    if patched > 0:
        tmp = in_json.with_suffix(".jsonl.tmp")
        with tmp.open("wb") as f:
            for rec in records:
                f.write(orjson.dumps(rec) + b"\n")
        tmp.replace(in_json)
        print(f"Patched {patched} records, wrote to {in_json}", file=sys.stderr)
    else:
        print("Nothing to patch.", file=sys.stderr)


if __name__ == "__main__":
    main()
