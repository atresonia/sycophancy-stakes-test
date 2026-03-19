import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
"""
Utility helper functions to manipulate input files and dataframes.
"""

def extract_extra_fields(
    row: pd.Series,
    col_names: Optional[List[str]] = None,
    key_to_col: Optional[Dict[str, str]] = None,
) -> dict:
    """Extract columns from a row into a dict. Default missing values to ''.

    Args:
        row: Series (e.g. a dataframe row).
        col_names: If provided, use these as both source column names and output keys.
        key_to_col: If provided, use as output_key -> source_column (overrides col_names).
    """
    if key_to_col is not None:
        return {out_key: row.get(source_col, "") for out_key, source_col in key_to_col.items()}
    if col_names is not None:
        return {col_name: row.get(col_name, "") for col_name in col_names}
    return {}


def load_df_from_jsonl(jsonl_path: Path, cols: List[str] = None, rename_cols: Dict[str, str] = None) -> pd.DataFrame:
    """Load a dataframe from a jsonl file."""
    df = pd.read_json(jsonl_path, lines=True)
    flat = pd.json_normalize(df.to_dict("records"), sep=".")
    if cols:
        flat = flat.loc[:, [c for c in cols if c in flat.columns]]
    if rename_cols:
        flat = flat.rename(columns=rename_cols)
    return flat

