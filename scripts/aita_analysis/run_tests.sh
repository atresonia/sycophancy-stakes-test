#!/usr/bin/env bash
# Run the test suite. Use from repo root: ./scripts/run_tests.sh
# Optional: pass pytest args, e.g. ./scripts/run_tests.sh -v tests/test_llm_inference.py
set -e
cd "$(dirname "$0")/.."
if ! python -c "import pytest" 2>/dev/null; then
  echo "Test deps missing. Install with: pip install -e '.[dev]' (or pip install -r requirements-dev.txt)"
  exit 1
fi
exec python -m pytest "$@" tests/
