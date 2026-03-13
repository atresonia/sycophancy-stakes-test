"""
Run the full pipeline with one example to verify scripts work before a full dataset run.

Uses temp or specified work dir; does not overwrite your real data.
Requires API key in env (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY).

Usage:
  python scripts/verify_pipeline.py (default: gemini-2.0-flash)
  python scripts/verify_pipeline.py --provider openai_compat --model gpt-5.2
  python scripts/verify_pipeline.py --provider anthropic --model claude-sonnet-4-20250514
  python scripts/verify_pipeline.py --work_dir ./verify_out --use_cache
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Project root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], step_name: str) -> bool:
    print(f"\n--- {step_name} ---")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"FAILED: {step_name} (exit code {result.returncode})", file=sys.stderr)
        return False
    return True


def _check_jsonl(path: Path, min_lines: int = 1, require_ok: bool = True) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Missing: {path}"
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < min_lines:
        return False, f"Expected at least {min_lines} line(s), got {len(lines)}"
    if require_ok:
        import json
        for i, ln in enumerate(lines):
            try:
                obj = json.loads(ln)
                if obj.get("status") != "ok":
                    return False, f"Line {i+1}: status != 'ok' ({obj.get('status')})"
            except json.JSONDecodeError as e:
                return False, f"Line {i+1}: invalid JSON ({e})"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pipeline with one example to verify scripts work.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=REPO_ROOT / "verify_out",
        help="Directory for intermediate and final outputs (created if needed)",
    )
    parser.add_argument(
        "--in_csv",
        type=Path,
        default=REPO_ROOT / "data" / "sample_one_aita.csv",
        help="Input CSV for step 1 (must have 'prompt' and 'ytanta' columns)",
    )
    parser.add_argument("--provider", type=str, choices=["openai_compat", "anthropic", "gemini"], default="gemini")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Model name (default varies by provider)")
    parser.add_argument("--use_cache", action="store_true", help="Use LLM cache for fresh calls")
    parser.add_argument(
        "--steps",
        type=str,
        default="1,2,3",
        help="Comma-separated steps to run (1=variants, 2=outputs, 3=syc_eval). Use e.g. '1' to only run step 1.",
    )
    args = parser.parse_args()

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    model = args.model
    in_csv = args.in_csv if args.in_csv.is_absolute() else REPO_ROOT / args.in_csv
    if not in_csv.exists():
        print(f"Input CSV not found: {in_csv}", file=sys.stderr)
        return 1

    steps_wanted = set()
    for s in args.steps.split(","):
        s = s.strip()
        if s in ("1", "2", "3"):
            steps_wanted.add(int(s))

    out1 = work_dir / "step1_stakes_variants.jsonl"
    out2 = work_dir / "step2_stakes_outputs.jsonl"
    out3 = work_dir / "step3_syc_eval.jsonl"

    common = ["--provider", args.provider, "--model", model, "--mode", "OVERWRITE"]
    if args.use_cache:
        common.append("--use_cache")

    # Step 1: Generate stakes variants (1 example)
    if 1 in steps_wanted:
        cmd1 = [
            sys.executable, str(REPO_ROOT / "scripts" / "generate_stakes_variants.py"),
            "--num_ex", "1",
            "--in_csv", str(in_csv),
            "--out_json", str(out1),
            *common,
        ]
        if not _run(cmd1, "Step 1: Generate stakes variants"):
            return 1
        ok, msg = _check_jsonl(out1, min_lines=1, require_ok=True)
        if not ok:
            print(f"Validation failed: {msg}", file=sys.stderr)
            return 1
        print(f"  Validated: 1 record, status=ok")

    # Step 2: Generate stakes outputs (YTA/NTA for low and high variants)
    if 2 in steps_wanted:
        if not out1.exists():
            print("Step 2 requires step 1 output. Run with --steps 1,2,3 or run step 1 first.", file=sys.stderr)
            return 1
        cmd2 = [
            sys.executable, str(REPO_ROOT / "scripts" / "generate_stakes_outputs.py"),
            "--in_json", str(out1),
            "--out_json", str(out2),
            "--num_ex", "1",
            *common,
        ]
        if not _run(cmd2, "Step 2: Generate stakes outputs"):
            return 1
        ok, msg = _check_jsonl(out2, min_lines=1, require_ok=True)
        if not ok:
            print(f"Validation failed: {msg}", file=sys.stderr)
            return 1
        print(f"  Validated: 1 record, status=ok")

    # Step 3: Sycophancy evaluation
    if 3 in steps_wanted:
        if not out2.exists():
            print("Step 3 requires step 2 output. Run with --steps 1,2,3 or run step 2 first.", file=sys.stderr)
            return 1
        cmd3 = [
            sys.executable, str(REPO_ROOT / "scripts" / "generate_syc_eval.py"),
            "--in_json", str(out2),
            "--out_json", str(out3),
            "--num_ex", "1",
            *common,
        ]
        if not _run(cmd3, "Step 3: Sycophancy eval"):
            return 1
        ok, msg = _check_jsonl(out3, min_lines=1, require_ok=True)
        if not ok:
            print(f"Validation failed: {msg}", file=sys.stderr)
            return 1
        print(f"  Validated: 1 record, status=ok")

    print("\n--- Verify pipeline OK ---")
    print(f"  Work dir: {work_dir}")
    if out1.exists():
        print(f"  Step 1 out: {out1}")
    if out2.exists():
        print(f"  Step 2 out: {out2}")
    if out3.exists():
        print(f"  Step 3 out: {out3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
