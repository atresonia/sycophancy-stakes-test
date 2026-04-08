"""JSON / structured-output parsing helpers for Gemini responses."""

import json
import re
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel


def _strip_json_from_text(text: str) -> str:
    """Strip optional markdown code fence so we can parse JSON from Gemini output."""
    text = text.strip()
    # ```json ... ``` or ``` ... ```
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _regex_extract_fields(text: str, schema: Type[BaseModel]) -> Optional[Dict[str, Any]]:
    """Last-resort field extraction from broken JSON using regex.

    Handles the common Gemini failure mode where the ``reasoning`` string value
    contains unescaped double-quote characters (e.g. the model quotes a phrase
    from the essay: ``"The author uses "vivid imagery"...``).  The ``grade``
    value — a single letter — is almost always intact, so we can recover it.

    For each field we try:
      1. A quoted-string match that stops at the first bare ``"`` inside the value
         (works for values with no inner quotes, e.g. ``"grade": "C"``).
      2. An unquoted-value match (handles ``"grade": C`` without quotes).
    """
    obj: Dict[str, Any] = {}
    for field_name in schema.model_fields:
        pattern_quoted = r'"' + re.escape(field_name) + r'"\s*:\s*"([^"]*)"'
        m = re.search(pattern_quoted, text)
        if m:
            obj[field_name] = m.group(1)
            continue
        pattern_bare = r'"' + re.escape(field_name) + r'"\s*:\s*([^,\}\s]+)'
        m = re.search(pattern_bare, text)
        if m:
            raw = m.group(1).strip('"')
            if raw == "null":
                obj[field_name] = None
            elif raw == "true":
                obj[field_name] = True
            elif raw == "false":
                obj[field_name] = False
            else:
                try:
                    obj[field_name] = int(raw)
                except ValueError:
                    try:
                        obj[field_name] = float(raw)
                    except ValueError:
                        obj[field_name] = raw
    return obj if obj else None


def _parse_structured_response(text: str, schema: Type[BaseModel]) -> Dict[str, Any]:
    """Parse JSON text into schema and return as dict. Ensures output conforms to schema."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as first_err:
        # Gemini occasionally emits control characters (e.g. raw newlines) in string
        # values that make the JSON invalid.  Strip C0/C1 control chars (keep \t\n\r)
        # and retry once before giving up.
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        try:
            obj = json.loads(sanitized)
        except json.JSONDecodeError:
            # Last resort: regex extraction for schemas with simple primary fields
            # (e.g. EssayGradeOutput where grade is a single letter and reasoning
            # may contain unescaped quotes that break JSON parsing).
            print(
                f"[Gemini JSON parse failure] schema={schema.__name__}, "
                f"error={first_err}, raw_text={repr(text[:300])}"
            )
            extracted = _regex_extract_fields(text, schema)
            if extracted:
                # Fill in any required str fields that regex couldn't recover (e.g. truncated reasoning)
                for fname, finfo in schema.model_fields.items():
                    if fname not in extracted and finfo.annotation is str:
                        extracted[fname] = ""
                try:
                    return schema.model_validate(extracted).model_dump()
                except Exception:
                    pass
            raise ValueError(
                f"Gemini returned invalid JSON for {schema.__name__}: {first_err}"
            ) from first_err
    try:
        return schema.model_validate(obj).model_dump()
    except Exception as e:
        raise ValueError(f"Gemini response did not match {schema.__name__} schema: {e}") from e
