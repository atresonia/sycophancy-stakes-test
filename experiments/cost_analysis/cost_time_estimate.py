"""
Cost and time estimator for the experiments.
"""

from typing import Callable
from dataclasses import dataclass
import tiktoken
import pandas as pd

PRICES = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
}

@dataclass
class Call:
    system_prompt: str | None
    user: str

def count(enc: tiktoken.Encoding, text: str) -> int:
    return len(enc.encode(text)) if text else 0


def estimate_run(
    df: pd.DataFrame,
    experiment_name: str,
    build_calls: Callable[[pd.Series], list[Call]],
    model: str,
    pilot_calls: int,
    pilot_seconds: float,
    max_concurrency: int = 16,
    max_output_tokens: int = 32,
) -> dict[str, float]:
    """Estimate the cost and time of running the experiment.
    Returns a dict of numbers
    Timing mode: the pilot processes `pilot_calls` calls in `pilot_seconds` seconds
    at some concurrency. Throughput is assumed to scale with the same concurrency.
    Rough projection
    """
    if model not in PRICES:
        raise ValueError(f"No price entry for model '{model}'. "
                         f"Known: {list(PRICES)}")
    
    enc = tiktoken.get_encoding("cl100k_base")

    n_rows = len(df)
    total_input = 0
    total_calls = 0
    calls_per_row = set()
    row_input_tokens = []

    for _, row in df.iterrows():
        calls = build_calls(row)
        calls_per_row.add(len(calls))
        row_total = 0
        for call in calls:
            row_total += count(enc, call.system_prompt) + count(enc, call.user)
        row_input_tokens.append(row_total)
        total_input += row_total
        total_calls += len(calls)
        row_input_tokens.append(row_total)
    
    total_output = total_calls * max_output_tokens
    
    p = PRICES[model]
    in_cost = total_input / 1e6 * p["input"]
    out_cost = total_output / 1e6 * p["output"]
    total_cost = in_cost + out_cost
 
    throughput = pilot_calls / pilot_seconds
    est_seconds = total_calls / throughput
 
    avg_row = sum(row_input_tokens) / len(row_input_tokens) if row_input_tokens else 0
    cpr = sorted(calls_per_row)
    cpr_str = str(cpr[0]) if len(cpr) == 1 else f"{cpr[0]}-{cpr[-1]} (varies)"
 
    print("=" * 64)
    print(f"RUN ESTIMATE ({experiment_name})")
    print("=" * 64)
    print(f"  Rows                  : {n_rows}")
    print(f"  Calls per row         : {cpr_str}")
    print(f"  Total API calls       : {total_calls:,}")
    print(f"  Model                 : {model}")
    print(f"  Avg input tokens/row  : {avg_row:.0f}")
    print("-" * 64)
    print(f"  Total input tokens    : {total_input:,} ({total_input/1e6:.2f}M)")
    print(f"  Total output tokens   : {total_output:,} ({total_output/1e6:.3f}M)"
          f"  [max_tokens cap]")
    print(f"  Input cost            : ${in_cost:.2f}")
    print(f"  Output cost           : ${out_cost:.2f}")
    print(f"  ESTIMATED TOTAL COST  : ${total_cost:.2f}")
    print("-" * 64)
    print(f"  Pilot throughput      : {throughput:.1f} calls/sec "
          f"({pilot_calls} calls / {pilot_seconds:.0f}s)")
    print(f"  ESTIMATED WALL TIME   : {est_seconds:.0f}s (~{est_seconds/60:.1f} min)")
    print("=" * 64)
    print("  NOTE: prices are May-2026 estimates and drift. Output uses the")
    print("  max_tokens cap (worst case); real output is smaller. Time assumes")
    print("  the same concurrency as the timing pilot.")
    print("=" * 64)
 
    return {
        "n_rows": n_rows, "total_calls": total_calls,
        "total_input_tokens": total_input, "total_output_tokens": total_output,
        "input_cost": in_cost, "output_cost": out_cost, "total_cost": total_cost,
        "est_seconds": est_seconds,
    }




def _essay_build_calls(row):
    from experiments.essay_grading_analysis.essay_grading_stakes import (
        LLM_ESSAY_GRADE_SYSTEM_PROMPT, STAKES_VARIANT_PROMPTS, USER_FRAMING_PREFIX,
    )
    sysp = LLM_ESSAY_GRADE_SYSTEM_PROMPT
    essay = str(row["essay"])
    calls = [Call(sysp, essay)]
    calls.append(Call(sysp, USER_FRAMING_PREFIX + "\n\n" + essay))
    for lvl in ("low", "medium", "high"):
        framing = USER_FRAMING_PREFIX + " " + STAKES_VARIANT_PROMPTS[lvl]
        calls.append(Call(sysp, framing + "\n\n" + essay))
    return calls
 
 
def estimate_essay_run(df, model, pilot_calls, pilot_seconds,
                       max_concurrency=16, max_output_tokens=32):
    return estimate_run(df, "essay grading stakes", _essay_build_calls, model,
                        pilot_calls, pilot_seconds, max_concurrency, max_output_tokens)
 
 
def _mmlu_build_calls(row):
    from experiments.mmlu_analysis.mmlu_sycophancy_base import (
        MMLU_BET_SYSTEM_PROMPT, MMLU_BET_USER_TEMPLATE, OPTION_LETTERS,
        build_conditions,
    )
    sysp = MMLU_BET_SYSTEM_PROMPT
    options = row["options"]
    options_block = "\n".join(
        f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options)
    )
    question_block = MMLU_BET_USER_TEMPLATE.format(
        question=row["question"], options_block=options_block,
    )
    conditions, _ = build_conditions(row)
    calls = []
    for _name, framing in conditions.items():
        user = f"{framing}\n\n{question_block}" if framing else question_block
        calls.append(Call(sysp, user))
    return calls
 
 
def estimate_mmlu_run(df, model, pilot_calls, pilot_seconds,
                      max_concurrency=16, max_output_tokens=32):
    return estimate_run(df, "mmlu bet", _mmlu_build_calls, model,
                        pilot_calls, pilot_seconds, max_concurrency, max_output_tokens)
 
 
def _elephant_build_calls(row, run_binary=True):
    from experiments.elephant_analysis.elephant_base_experiment_flip import STAKE_VARIANTS
    suffix = "\nOutput only YTA or NTA." if run_binary else ""
    orig = str(row["original_post"])
    flip = str(row["flipped_story"])
    calls = [Call(None, orig + suffix), Call(None, flip + suffix)]
    for lvl in ("low", "medium", "high"):
        v = STAKE_VARIANTS[lvl]
        calls.append(Call(None, orig + "\n\n" + v + suffix))
        calls.append(Call(None, flip + "\n\n" + v + suffix))
    return calls
 
 
def estimate_elephant_run(df, model, pilot_calls, pilot_seconds,
                          max_concurrency=16, max_output_tokens=32,
                          run_binary=True):
    bc = lambda row: _elephant_build_calls(row, run_binary=run_binary)
    return estimate_run(df, "elephant", bc, model,
                        pilot_calls, pilot_seconds, max_concurrency, max_output_tokens)