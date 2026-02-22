import pandas as pd
from pathlib import Path
from typing import List, Dict
"""
Utility helper functions to manipulate input files and dataframes.
"""

def load_df_from_jsonl(jsonl_path: Path, cols: List[str] = None, rename_cols: Dict[str, str] = None) -> pd.DataFrame:
    """Load a dataframe from a jsonl file."""
    df = pd.read_json(jsonl_path, lines=True)
    flat = pd.json_normalize(df.to_dict("records"), sep=".")
    if cols:
        flat = flat.loc[:, [c for c in cols if c in flat.columns]]
    if rename_cols:
        flat = flat.rename(columns=rename_cols)
    return flat