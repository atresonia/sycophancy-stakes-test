"""MMLU baseline-sycophancy experiment, routed through inference/ infrastructure.

Same prompts and output schema as mmlu_sycophancy_base.py, but routed through
LLMClient: rows + probes run concurrently bounded by a semaphore, with retries
on rate-limit errors and optional disk caching.

Output is a flat DataFrame (one row per question) — same six columns as v1.
"""
import argparse
import asyncio
import os
import random
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from datasets import load_dataset

from inference.client import EndpointConfig, LLMClient
from inference.task import InferenceTask
from scripts.mmlu_analysis.mmlu_sycophancy_base import (
    KNOWLEDGE_BASELINE_SYSTEM_PROMPT,
    KNOWLEDGE_BASELINE_USER_TEMPLATE,
    USER_SYCOPHANCY_BASELINE_USER_TEMPLATE,
    extract_answer_letter,
)

OPTION_LETTERS = string.ascii_uppercase  # A-Z


def load_mmlu_pro(num_ex: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Same as v1's loader but uses to_list() instead of to_pandas() to avoid
    pyarrow's 'Wrong number of dimensions' error on list columns.
    """
    mmlu_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    mmlu_df = pd.DataFrame(mmlu_ds.to_list())
    mmlu_df = mmlu_df.assign(
        correct_answer=mmlu_df.apply(lambda r: r["options"][int(r["answer_index"])], axis=1),
        correct_answer_letter=mmlu_df.apply(
            lambda r: OPTION_LETTERS[int(r["answer_index"])], axis=1,
        ),
        incorrect_answers=mmlu_df.apply(
            lambda r: [o for i, o in enumerate(r["options"]) if i != int(r["answer_index"])],
            axis=1,
        ),
        incorrect_answers_letters=mmlu_df.apply(
            lambda r: [
                OPTION_LETTERS[i] for i, _ in enumerate(r["options"]) if i != int(r["answer_index"])
            ],
            axis=1,
        ),
    )
    if num_ex is not None:
        mmlu_df = mmlu_df.sample(n=num_ex, random_state=seed)
    return mmlu_df


def _build_kb_task(row: pd.Series, probe_idx: int, max_tokens: int) -> InferenceTask:
    options = row["options"]
    options_block = "\n".join(
        f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)
    )
    letter_options = ", ".join(OPTION_LETTERS[: len(options)])
    user_prompt = KNOWLEDGE_BASELINE_USER_TEMPLATE.format(
        question=row["question"],
        options_block=options_block,
    )
    system = (
        KNOWLEDGE_BASELINE_SYSTEM_PROMPT
        + "Respond with the letter only "
        + letter_options
    )
    return InferenceTask(
        user_prompt=user_prompt,
        system_prompt=system,
        temperature=1.0,
        max_tokens=max_tokens,
        thinking_budget=0,
        # repeat_idx differentiates cache keys across the num_probes calls so
        # each probe is treated as a distinct sample under a warm cache.
        repeat_idx=probe_idx,
    )


def _build_syc_task(
    row: pd.Series,
    seed: int,
    max_tokens: int,
) -> Tuple[InferenceTask, str, str]:
    options = row["options"]
    options_block = "\n".join(
        f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)
    )
    letter_options = ", ".join(OPTION_LETTERS[: len(options)])
    rng = random.Random(seed)
    user_answer = rng.choice(row["incorrect_answers_letters"])

    user_prompt = USER_SYCOPHANCY_BASELINE_USER_TEMPLATE.format(
        question=row["question"],
        options_block=options_block,
        user_answer=user_answer,
        letter_options=letter_options,
    )
    full_prompt = user_prompt + "Respond with the letter only " + letter_options
    task = InferenceTask(
        user_prompt=full_prompt,
        temperature=1.0,
        max_tokens=max_tokens,
        thinking_budget=0,
    )
    # Return the un-suffixed prompt so the output column matches v1.
    return task, user_prompt, user_answer


async def run_sycophancy_base_experiment(
    df: pd.DataFrame,
    num_probes: int = 10,
    model: str = "gemini-2.5-flash",
    max_kb_tokens: int = 64,
    max_syc_tokens: int = 512,
    seed: int = 42,
    max_concurrency: int = 16,
    api_key: str | None = None,
    use_cache: bool = False,
) -> pd.DataFrame:
    """Run kb probes + user-sycophancy probe for every row, fully concurrent.

    Returns a flat DataFrame with one row per question:
        question_id, prompt, user_answer, user_syc_llm_answer,
        correct_answer, llm_kb_answer
    """
    client = LLMClient(
        endpoint=EndpointConfig(
            provider="gemini",
            api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
            model=model,
        ),
        max_concurrency=max_concurrency,
        cache_dir=".llm_cache" if use_cache else None,
    )

    async def process_row(idx: int, row: pd.Series) -> Dict[str, Any]:
        kb_tasks = [
            _build_kb_task(row, probe_idx=p, max_tokens=max_kb_tokens)
            for p in range(num_probes)
        ]
        syc_task, syc_user_prompt, user_answer = _build_syc_task(
            row, seed=seed + idx, max_tokens=max_syc_tokens
        )

        results = await asyncio.gather(
            *(client.run(t) for t in kb_tasks),
            client.run(syc_task),
        )
        kb_results, syc_result = results[:-1], results[-1]
        n_options = len(row["options"])

        kb_answers = []
        for r in kb_results:
            ans = extract_answer_letter(r["response_content"], n_options)
            if ans is None:
                print(
                    f"knowledge_baseline extract answer_letter failed: "
                    f"could not extract from {r['response_content']!r}"
                )
            else:
                kb_answers.append(ans)
        kb_majority = (
            Counter(kb_answers).most_common(1)[0][0] if kb_answers else None
        )

        syc_text = syc_result["response_content"]
        syc_answer = extract_answer_letter(syc_text, n_options)
        if syc_answer is None:
            print(
                f"user_sycophancy_baseline extract answer_letter failed: "
                f"could not extract from {syc_text!r}"
            )

        return {
            "question_id": row["question_id"],
            "prompt": syc_user_prompt,
            "user_answer": user_answer,
            "user_syc_llm_answer": syc_answer,
            "correct_answer": row["correct_answer_letter"],
            "llm_kb_answer": kb_majority,
        }

    rows = await asyncio.gather(
        *(process_row(i, r) for i, (_, r) in enumerate(df.iterrows()))
    )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMLU baseline-sycophancy (LLMClient).")
    p.add_argument("--num_ex", type=int, default=100)
    p.add_argument("--num_probes", type=int, default=10)
    p.add_argument("--model", type=str, default="gemini-2.5-flash")
    p.add_argument("--max_kb_tokens", type=int, default=64)
    p.add_argument("--max_syc_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_concurrency", type=int, default=16)
    p.add_argument("--use_cache", action="store_true", help="Enable disk cache at .llm_cache/")
    p.add_argument(
        "--out_csv", type=Path,
        default=Path("data/mmlu/mmlu_sycophancy_base_v2.csv"),
    )
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()
    df = load_mmlu_pro(num_ex=args.num_ex, seed=args.seed)
    print(f"Loaded {len(df)} MMLU questions.")

    t0 = time.perf_counter()
    out = await run_sycophancy_base_experiment(
        df,
        num_probes=args.num_probes,
        model=args.model,
        max_kb_tokens=args.max_kb_tokens,
        max_syc_tokens=args.max_syc_tokens,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        use_cache=args.use_cache,
    )
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s ({len(out)} rows).")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    asyncio.run(_main())
