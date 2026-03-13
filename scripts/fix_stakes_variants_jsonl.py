"""
Normalize stakes-variants JSONL so every "ok" row has output conforming to StakesOutput schema.

- If output is a string (raw JSON), parse and validate, then replace with validated dict.
- If output is a dict, validate with StakesOutput and replace with model_dump() for consistency.
- Rows with status "error" are left unchanged.
- Rows that cannot be normalized are optionally marked as error (--strict) or skipped (default).

Usage:
  python scripts/fix_stakes_variants_jsonl.py --in_json data/gemini/aita-yta-stakes-variants.jsonl --out_json data/gemini/aita-yta-stakes-variants.jsonl
  python scripts/fix_stakes_variants_jsonl.py --in_json data/gemini/aita-yta-stakes-variants.jsonl  # writes in place if --out_json omitted
"""
import argparse
import json
import re
from pathlib import Path

from pydantic import ValidationError

from inference.llm_inference import StakesOutput


def _strip_json_code_block(text: str) -> str:
    """Remove optional markdown code fence around JSON."""
    text = text.strip()
    # ```json ... ``` or ``` ... ```
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def normalize_output(output: object) -> dict:
    """
    Normalize output to a dict that conforms to StakesOutput.
    Raises ValueError or ValidationError if not possible.
    """
    if isinstance(output, dict):
        raw = output
    elif isinstance(output, str):
        raw = json.loads(_strip_json_code_block(output))
    else:
        raise ValueError(f"output must be dict or JSON string, got {type(output)}")
    return StakesOutput.model_validate(raw).model_dump()


def main():
    parser = argparse.ArgumentParser(description="Normalize stakes variants JSONL to StakesOutput schema.")
    parser.add_argument("--in_json", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSONL path (default: overwrite in_json)")
    parser.add_argument("--strict", action="store_true", help="Mark unparseable 'ok' rows as error instead of keeping as-is")
    args = parser.parse_args()

    in_path = Path(args.in_json)
    out_path = Path(args.out_json) if args.out_json else in_path
    in_path = in_path.resolve()
    out_path = out_path.resolve()

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    records = []
    fixed = 0
    broken = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                records.append(rec)
                continue
            out = rec.get("output")
            if out is None:
                records.append(rec)
                continue
            try:
                rec["output"] = normalize_output(out)
                fixed += 1
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                if args.strict:
                    rec["status"] = "error"
                    rec["error"] = f"normalization failed: {e}"
                    if "output" in rec:
                        del rec["output"]
                    broken += 1
                # else: keep record as-is
            records.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path}. Normalized {fixed} outputs." + (f" Marked {broken} as error (--strict)." if broken else ""))


if __name__ == "__main__":
    main()
