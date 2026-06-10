"""
Cost and time estimator for the experiments.
"""

from typing import Callable
from dataclasses import dataclass
import tiktoken
import pandas as pd

PRICES = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 5.00},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "us.meta.llama3-1-8b-instruct-v1:0": {"input": 0.22, "output": 0.22}, # https://aws.amazon.com/bedrock/pricing/
    "mistral.ministral-3-14b-instruct": {"input": 0.15, "output": 0.15},
    "openai.gpt-oss-20b-1:0": {"input": 0.07, "output": 0.30},
    "openai.gpt-oss-120b-1:0": {"input": 0.15, "output": 0.60},
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
        print(f"No price entry for model '{model}'. Known: {list(PRICES)}")
        return {
            "n_rows": len(df),
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0,
            "est_seconds": 0,
        }
    
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
 
    return {
        "n_rows": n_rows, "total_calls": total_calls,
        "total_input_tokens": total_input, "total_output_tokens": total_output,
        "input_cost": in_cost, "output_cost": out_cost, "total_cost": total_cost,
        "est_seconds": est_seconds,
    }


def essay_build_calls_scenarios(row: pd.Series, column: str = "essay") -> list[Call]:
    from experiments.essay_grading_analysis.essay_grading_scenarios import (
        SCENARIOS,
        SCENARIOS_PROMPT_TEMPLATE,
        LEVELS,
    )

    calls = []
    # extract essay from "kb_prompt"
    essay = row["kb_prompt"].split("\n\n")[0]
    # 1. recursively find "prompt" in SCENARIOS
    # note: it is either 
    # a) directly under the scenario key or 
    # b) under the keys under the scenario key (one of: low, medium, high, submitted, not_submitted)
    for scenario in SCENARIOS:
        if "prompt" in SCENARIOS[scenario]:
            calls.append(Call(
                system_prompt=None, 
                user=SCENARIOS_PROMPT_TEMPLATE.format(essay=essay, scenario_prompt=SCENARIOS[scenario]["prompt"])
            ))
        else:
            for level in LEVELS:
                if level in SCENARIOS[scenario] and "prompt" in SCENARIOS[scenario][level]:
                    calls.append(Call(
                        system_prompt=None, 
                        user=SCENARIOS_PROMPT_TEMPLATE.format(essay=essay, scenario_prompt=SCENARIOS[scenario][level]["prompt"])
                    ))
    return calls



def essay_build_calls(row: pd.Series, column: str = "essay") -> list[Call]:
    from experiments.essay_grading_analysis.essay_grading_stakes import (
        STAKES_VARIANT_PROMPTS, USER_FRAMING_PREFIX,
    )

    essay = str(row[column])
    calls = [Call(system_prompt=None, user=essay)]
    calls.append(Call(system_prompt=None, user=USER_FRAMING_PREFIX + "\n\n" + essay))
    for level in ["low", "medium", "high"]:
        user_prompt = USER_FRAMING_PREFIX + " " + STAKES_VARIANT_PROMPTS[level] + "\n\n" + essay
        calls.append(Call(system_prompt=None, user=user_prompt))
    return calls


def estimate_essay_scenarios_run(
    df: pd.DataFrame, 
    model: str, 
    pilot_calls: int, 
    pilot_seconds: float, 
    max_concurrency: int = 16, 
    max_output_tokens: int = 32,
    column: str = "essay",
) -> dict[str, float]:

    return estimate_run(
        df=df, 
        experiment_name="essay grading scenarios", 
        build_calls=lambda row: essay_build_calls_scenarios(row, column), 
        model=model,
        pilot_calls=pilot_calls, 
        pilot_seconds=pilot_seconds, 
        max_concurrency=max_concurrency, 
        max_output_tokens=max_output_tokens
    )


def estimate_essay_run(
    df: pd.DataFrame, 
    model: str, 
    pilot_calls: int, 
    pilot_seconds: float, 
    max_concurrency: int = 16, 
    max_output_tokens: int = 32,
    column: str = "essay",
) -> dict[str, float]:
    return estimate_run(
        df=df, 
        experiment_name="essay grading stakes", 
        build_calls=lambda row: essay_build_calls(row, column), 
        model=model,
        pilot_calls=pilot_calls, 
        pilot_seconds=pilot_seconds, 
        max_concurrency=max_concurrency, 
        max_output_tokens=max_output_tokens
    )


def mmlu_build_calls(row: pd.Series) -> list[Call]:
    from experiments.mmlu_analysis.mmlu_sycophancy_base import (
        MMLU_BET_USER_TEMPLATE, OPTION_LETTERS, build_conditions
    )

    options = row["options"]
    options_block = "\n".join(
        f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options)
    )
    question_block = MMLU_BET_USER_TEMPLATE.format(
        question=row["question"], options_block=options_block,
    )
    conditions, _ = build_conditions(row)
    calls = []
    for _, framing in conditions.items():
        user_prompt = f"{framing}\n\n{question_block}" if framing else question_block
        calls.append(Call(system_prompt=None, user=user_prompt))
    return calls


def estimate_mmlu_run(
    df: pd.DataFrame, 
    model: str, 
    pilot_calls: int, 
    pilot_seconds: float, 
    max_concurrency: int = 16, 
    max_output_tokens: int = 32,
    column: str = "essay",
) -> dict[str, float]:
    return estimate_run(
        df=df, 
        experiment_name="mmlu bet", 
        build_calls=mmlu_build_calls, 
        model=model, 
        column=column,
        pilot_calls=pilot_calls, 
        pilot_seconds=pilot_seconds, 
        max_concurrency=max_concurrency, 
        max_output_tokens=max_output_tokens
    )


def elephant_build_calls(row: pd.Series, run_binary: bool = True) -> list[Call]:
    from experiments.elephant_analysis.elephant_base_experiment_flip import (
        STAKE_VARIANTS,
    )

    suffix = "\nOutput only YTA or NTA." if run_binary else ""
    original = str(row["original_post"])
    flipped = str(row["flipped_story"])
    calls = [Call(system_prompt=None, user=original + suffix), Call(system_prompt=None, user=flipped + suffix)]
    for level in ["low", "medium", "high"]:
        original_user_prompt = STAKE_VARIANTS[level] + "\n\n" + original + suffix
        calls.append(Call(system_prompt=None, user=original_user_prompt))
        flipped_user_prompt = STAKE_VARIANTS[level] + "\n\n" + flipped + suffix
        calls.append(Call(system_prompt=None, user=flipped_user_prompt))
    return calls

def estimate_elephant_run(
    df: pd.DataFrame, model: str, pilot_calls: int, 
    pilot_seconds: float, max_concurrency: int = 16, max_output_tokens: int = 32
) -> dict[str, float]:
    return estimate_run(
        df=df, experiment_name="elephant", 
        build_calls=elephant_build_calls, 
        model=model, 
        pilot_calls=pilot_calls,
        pilot_seconds=pilot_seconds,
        max_concurrency=max_concurrency,
        max_output_tokens=max_output_tokens
    )