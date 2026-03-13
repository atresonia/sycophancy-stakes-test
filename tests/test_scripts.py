"""Tests for script entry points (argparse and wiring, no real API calls)."""
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pandas as pd
import pytest


def test_generate_stakes_variants_help() -> None:
    """Script parses args and prints help."""
    with patch.object(sys, "argv", ["script", "--help"]):
        with pytest.raises(SystemExit):
            from scripts.generate_stakes_variants import main
            main()


def test_generate_stakes_variants_run_with_mock(
    tmp_path: Path,
    sample_aita_row: dict,
) -> None:
    """Run generate_stakes_variants with 1 row and mocked client (no API key)."""
    csv_path = tmp_path / "aita.csv"
    pd.DataFrame([sample_aita_row]).to_csv(csv_path, index=False)
    out_path = tmp_path / "out.jsonl"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
        with patch(
            "scripts.generate_stakes_variants.run_batch_inference",
            new_callable=AsyncMock,
        ) as mock_batch:
            with patch("scripts.generate_stakes_variants.LLMClient"):
                with patch.object(sys, "argv", [
                    "script",
                    "--num_ex", "1",
                    "--model", "gpt-test",
                    "--in_csv", str(csv_path),
                    "--out_json", str(out_path),
                ]):
                    from scripts.generate_stakes_variants import main
                    main()
            mock_batch.assert_called_once()
            call_kw = mock_batch.call_args[1]
            assert call_kw["out_json"] == out_path
            assert call_kw["n"] == 1


def test_generate_syc_eval_help() -> None:
    with patch.object(sys, "argv", ["script", "--help"]):
        with pytest.raises(SystemExit):
            from scripts.generate_syc_eval import main
            main()


def test_generate_stakes_outputs_help() -> None:
    with patch.object(sys, "argv", ["script", "--help"]):
        with pytest.raises(SystemExit):
            from scripts.generate_stakes_outputs import main
            main()
