"""MMLU bet-stakes experiment with a two-pass design.

Pass 1 (knowledge baseline, stochastic temperature=1.0): each question is asked
--kb_repeats times in 10-way MC format with no social/bet framing. Aggregated
answer counts identify the model-specific hardest distractor per question.

Pass 2 (bet-stakes, deterministic --temperature, default 0.0): each question is
run through 40 cells (10 stakes variants x 2 user_stances x 2 positionals) via
create_mmlu_bet_task using the hardest distractor from Pass 1.

Usage:
    # Full two-pass run:
    python scripts/mmlu_analysis/generate_mmlu_bet_stakes.py \\
        --num_ex 500 --model gemini-2.5-flash --provider gemini \\
        --out_json data/mmlu/mmlu_bet_stakes_gemini.jsonl

    # Skip Pass 1 if knowledge baseline already saved:
    python scripts/mmlu_analysis/generate_mmlu_bet_stakes.py \\
        --num_ex 500 --model gemini-2.5-flash --provider gemini \\
        --out_json data/mmlu/mmlu_bet_stakes_gemini.jsonl \\
        --kb_json data/mmlu/mmlu_bet_stakes_gemini_kb.jsonl
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import orjson
import pandas as pd

from data_loading.mmlu import (
    load_mmlu_pro, select_easy_distractor, select_hardest_distractor,
)
from inference.batch_inference import RunMode, run_batch_inference, run_multi_task_inference
from inference.client import EndpointConfig, LLMClient
from inference.prompts.mmlu import (
    MMLU_STAKES_VARIANTS, POSITIONALS, USER_STANCE,
    create_mmlu_bet_task, create_mmlu_knowledge_baseline_task,
)
from inference.schemas.truthfulqa import BetAnswerOutput, parse_one_word_response
from inference.task import InferenceTask
from scripts.utility_analysis.experiment_runner import build_parser, resolve_api_key

LETTERS = "ABCDEFGHIJ"
_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])")


def expand_with_repeats(df: pd.DataFrame, m_repeats: int) -> pd.DataFrame:
    expanded = df.merge(pd.DataFrame({"repeat_idx": range(m_repeats)}), how="cross")
    expanded.index = expanded["question_id"] * m_repeats + expanded["repeat_idx"]
    expanded.index.name = None
    return expanded


async def run_knowledge_baseline(
    df: pd.DataFrame, client: LLMClient, kb_repeats: int, kb_json: Path,
) -> dict[int, dict[str, int]]:
    """Pass 1. Returns {question_id: {letter: pick_count}}; saves to kb_json.
    If kb_json already exists, loads and returns without calling the API."""
    if kb_json.exists():
        with kb_json.open("rb") as f:
            results = {int((o := orjson.loads(line))["question_id"]): o["answer_counts"] for line in f}
        print(f"Loaded knowledge baseline from {kb_json}")
        return results

    def factory(row: pd.Series) -> InferenceTask:
        task = create_mmlu_knowledge_baseline_task(
            question=row["question"], options=list(row["options"]), temperature=1.0,
        )
        task.repeat_idx = int(row["repeat_idx"])
        return task

    expanded = expand_with_repeats(df, kb_repeats)
    records = await run_batch_inference(
        df=expanded, client=client, task_factory=factory,
        n=len(expanded), mode=RunMode.OVERWRITE,
        extra_fields=lambda r: {"question_id": int(r["question_id"])},
    )

    qid_to_correct = {int(r["question_id"]): LETTERS[int(r["answer_index"])]
                      for _, r in df.iterrows()}
    qid_to_prompt = {
        int(r["question_id"]): create_mmlu_knowledge_baseline_task(
            question=r["question"], options=list(r["options"]),
        ).get_user_prompt()
        for _, r in df.iterrows()
    }
    counts: dict[int, Counter] = {}
    for rec in records.values():
        if rec.get("status") != "ok":
            continue
        m = _LETTER_RE.search(str(rec["output"]).strip().upper())
        if m:
            counts.setdefault(int(rec["question_id"]), Counter())[m.group(1)] += 1

    kb_json.parent.mkdir(parents=True, exist_ok=True)
    with kb_json.open("wb") as f:
        for qid in sorted(counts):
            ac = dict(counts[qid])
            f.write(orjson.dumps({
                "question_id": qid, "answer_counts": ac,
                "correct_answer_letter": qid_to_correct[qid],
                "is_correct_majority": max(ac, key=ac.get) == qid_to_correct[qid],
                "prompt": qid_to_prompt[qid],
            }) + b"\n")
    print(f"Saved knowledge baseline to {kb_json}")
    return {qid: dict(c) for qid, c in counts.items()}


DISTRACTOR_PLAUSIBILITY: tuple[str, str] = ("hard", "easy")


def build_mmlu_task_factories(
    variants: Dict[str, str], temperature: float = 0.0,
) -> Dict[str, Callable[[pd.Series], InferenceTask]]:
    """Build 80 bet factories: 10 variants x 2 stances x 2 positionals x 2 distractors."""
    factories: Dict[str, Callable[[pd.Series], InferenceTask]] = {}
    for vname, premise in variants.items():
        for stance in USER_STANCE:
            for pos in POSITIONALS:
                for distractor in DISTRACTOR_PLAUSIBILITY:
                    def factory(row: pd.Series, _p=premise, _s=stance, _o=pos, _d=distractor):
                        inc = row["incorrect_answer"] if _d == "hard" else row["incorrect_answer_easy"]
                        return create_mmlu_bet_task(
                            question=row["question"],
                            correct_answer=row["correct_answer"],
                            incorrect_answer=inc,
                            premise=_p, positional=_o, user_stance=_s, temperature=temperature,
                        )
                    factories[f"{vname}_{stance}_{pos}_{distractor}"] = factory
    return factories


def make_combine_outputs(variants: Dict[str, str]):
    """Assemble the 80 per-key outputs into a single output dict per row."""
    meta = {f"{v}_{s}_{p}_{d}": (v, s, p, d)
            for v in variants for s in USER_STANCE for p in POSITIONALS
            for d in DISTRACTOR_PLAUSIBILITY}

    def combine(outputs: Dict[str, Any], base: Dict[str, Any],
                prompts: Dict[str, str]) -> Dict[str, Any]:
        cells = []
        for key, raw in outputs.items():
            v, s, p, d = meta[key.removesuffix("_output")]
            raw_text = raw if isinstance(raw, str) else str(raw)
            cells.append({
                "variant": v,
                "positional": p,
                "user_stance": s,
                "distractor_plausibility": d,
                "prompt": prompts.get(key, ""),
                "response": BetAnswerOutput(
                    answer=parse_one_word_response(raw_text),
                    raw_response=raw_text,
                ).model_dump(),
            })
        return {"output": {k: base[k] for k in
                ("question_id",
                "correct_answer",
                "incorrect_answer",
                "incorrect_answer_easy",
                "repeat_idx")}
                | {"cells": cells}}
    return combine


def main() -> None:
    parser = build_parser("Generate MMLU bet-stakes responses (two-pass)")
    parser.add_argument("--m_repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--kb_repeats", type=int, default=30)
    parser.add_argument("--kb_json", type=str, default=None)
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key_env, args.provider)
    cleanup: list[Path] = []
    out_path = Path(args.out_json) if args.out_json else Path(
        tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name)
    if not args.out_json:
        cleanup.append(out_path)
    if args.kb_json:
        kb_path = Path(args.kb_json)
    elif args.out_json:
        kb_path = out_path.with_name(out_path.stem + "_kb.jsonl")
    else:
        kb_path = Path(tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name)
        cleanup.append(kb_path)

    df = load_mmlu_pro(num_ex=args.num_ex, seed=1234)
    make_client = lambda: LLMClient(
        endpoint=EndpointConfig(provider=args.provider, model=args.model,
                                base_url=args.base_url, api_key=api_key),
        cache_dir=None, max_concurrency=args.max_concurrency,
    )

    try:
        kb = asyncio.run(run_knowledge_baseline(df, make_client(), args.kb_repeats, kb_path))
        n_before = len(df)
        keep_mask = df.apply(
            lambda r: int(r["question_id"]) in kb
            and max(kb[int(r["question_id"])], key=kb[int(r["question_id"])].get)
            == LETTERS[int(r["answer_index"])],
            axis=1,
        )
        df = df[keep_mask].reset_index(drop=True)
        print(f"[filter] kept {len(df)} / {n_before} questions where KB majority == correct answer")
        df = select_hardest_distractor(kb, df)
        df = select_easy_distractor(kb, df, np.random.default_rng(1234))
        df_bet = expand_with_repeats(df, args.m_repeats)

        def extra_fields_fn(row: pd.Series) -> Dict[str, Any]:
            return {
                "question_id": int(row["question_id"]),
                "correct_answer": row["correct_answer"],
                "incorrect_answer": row["incorrect_answer"],
                "incorrect_answer_easy": row["incorrect_answer_easy"],
                "repeat_idx": int(row["repeat_idx"]),
                "kb_answer_counts": kb.get(int(row["question_id"]), {}),
            }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(run_multi_task_inference(
            df=df_bet, client=make_client(),
            task_factories=build_mmlu_task_factories(MMLU_STAKES_VARIANTS, args.temperature),
            out_json=out_path, n=len(df_bet), mode=args.mode,
            force_ids=[int(x) for x in args.force_ids.split(",")] if args.force_ids else None,
            extra_fields=extra_fields_fn,
            combine_outputs=make_combine_outputs(MMLU_STAKES_VARIANTS),
        ))
        if not args.out_json:
            print(out_path.read_text(), end="")
    finally:
        for p in cleanup:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
