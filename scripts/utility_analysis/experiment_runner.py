"""Shared infrastructure for essay grading experiment scripts."""
import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from inference.batch_inference import run_multi_task_inference, RunMode
from inference.client import EndpointConfig, LLMClient
from inference.prompts.essay import create_llm_essay_grade_task
from inference.task import InferenceTask
from utils.helpers import extract_extra_fields


USER_FRAMING_PREFIX = "I wrote this essay. Please grade it."


def format_user_framing(stakes_text: str) -> str:
    if stakes_text:
        return f"{USER_FRAMING_PREFIX} {stakes_text}"
    return USER_FRAMING_PREFIX


def make_task_factory(variant_text: str):
    """Return a task factory function for a given variant framing text."""
    def factory(row: pd.Series):
        return create_llm_essay_grade_task(
            user_framing=format_user_framing(variant_text),
            prompt=row["essay"],
            temperature=0.7,
        )
    return factory


def resolve_api_key(api_key_env: Optional[str], provider: str) -> str:
    if api_key_env:
        env_value = os.getenv(api_key_env)
        if env_value:
            return env_value
        raise ValueError(
            f"--api_key_env was set to '{api_key_env}', but that environment variable is not set."
        )
    provider_env_fallbacks: Dict[str, List[str]] = {
        "openai_compat": ["OPENAI_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "gemini": ["GEMINI_API_KEY"],
    }
    env_names = provider_env_fallbacks.get(provider, [])
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
    raise ValueError(f"No API key found. Set --api_key_env, or one of: {', '.join(env_names)}")


def build_parser(description: str = "Run essay grading experiment") -> argparse.ArgumentParser:
    """Build the standard CLI parser shared by all experiment scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num_ex", type=int, default=None)
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--provider", type=str, choices=["openai_compat", "anthropic", "gemini"], default="openai_compat")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default=None,
                        help="Environment variable name that stores the API key.")
    parser.add_argument("--in_csv", type=str, default="data/essay_grading/lower_B_C_D_F_grade_df.csv")
    parser.add_argument("--out_json", type=str, default=None,
                        help="Path to save output JSONL. If omitted, prints to stdout.")
    parser.add_argument("--mode", type=lambda s: RunMode[s.upper()], choices=list(RunMode),
                        default=RunMode.OVERWRITE)
    parser.add_argument("--force_ids", type=str, default=None)
    parser.add_argument("--use_cache", action="store_true", help="Use LLM response caching")
    parser.add_argument("--max_concurrency", type=int, default=None,
                        help="Max concurrent LLM requests. Uses provider-appropriate default if unset.")
    return parser


def run_experiment(
    variants: Dict[str, str],
    args: argparse.Namespace,
    *,
    load_data: Callable[[argparse.Namespace], pd.DataFrame] | None = None,
    build_task_factories: Callable[[Dict[str, str]], Dict[str, Callable[[pd.Series], InferenceTask]]] | None = None,
    extra_fields_fn: Callable[[pd.Series], Dict[str, Any]] | None = None,
    combine_outputs: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]] | None = None,
) -> None:
    """Run an experiment with the given variant dict and parsed CLI args.

    When the optional callbacks are ``None``, falls back to essay-specific
    defaults (CSV loading, essay task factories, essay extra fields).

    Args:
        variants: Mapping of variant name to variant text/config.
        args: Parsed CLI namespace from ``build_parser``.
        load_data: Custom data loader. Receives ``args``, returns a DataFrame
            whose index is used as ``prompt_row_id``.
        build_task_factories: Custom factory builder. Receives ``variants``,
            returns ``{name: row -> InferenceTask}`` dict.
        extra_fields_fn: Custom extra-field extractor per row.
        combine_outputs: Callback passed to ``run_multi_task_inference`` to
            reshape per-key outputs into a combined record.
    """
    force_ids = [int(fid) for fid in args.force_ids.split(",")] if args.force_ids else None
    num_ex = args.num_ex
    api_key = resolve_api_key(api_key_env=args.api_key_env, provider=args.provider)

    print(f"Generating variant grades for {args.in_csv} with {num_ex} examples "
          f"using {args.model} in {args.mode.name} mode...", file=sys.stderr)

    # --- Data loading ---
    if load_data is not None:
        df = load_data(args)
        num_ex = len(df)
    else:
        df = pd.read_csv(args.in_csv)
        if num_ex > len(df):
            print(f"Warning: num_ex ({num_ex}) > rows ({len(df)}). Using {len(df)}.", file=sys.stderr)
            num_ex = len(df)
        df = df.sample(n=num_ex, random_state=1234).set_index("essay_id")

    client = LLMClient(
        endpoint=EndpointConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
        ),
        cache_dir=".llm_cache" if args.use_cache else None,
        max_concurrency=args.max_concurrency,
    )

    # --- Task factories ---
    if build_task_factories is not None:
        task_factories = build_task_factories(variants)
    else:
        task_factories = {name: make_task_factory(text) for name, text in variants.items()}

    # --- Extra fields ---
    if extra_fields_fn is None:
        extra_fields_fn = lambda row: extract_extra_fields(row, col_names=["ground_truth", "llm_response"])

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(run_multi_task_inference(
            df=df, client=client, task_factories=task_factories,
            out_json=out_path, n=num_ex, mode=args.mode, force_ids=force_ids,
            extra_fields=extra_fields_fn,
            combine_outputs=combine_outputs,
        ))
    else:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            asyncio.run(run_multi_task_inference(
                df=df, client=client, task_factories=task_factories,
                out_json=tmp_path, n=num_ex, mode=args.mode, force_ids=force_ids,
                extra_fields=extra_fields_fn,
                combine_outputs=combine_outputs,
            ))
            print(tmp_path.read_text(), end="")
        finally:
            tmp_path.unlink(missing_ok=True)
