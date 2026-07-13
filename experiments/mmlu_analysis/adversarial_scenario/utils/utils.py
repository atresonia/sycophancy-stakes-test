import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_json(
    file_path: str,
    row_keys: list[str] = ["prompt", "tag", "directional", "fr"],
    num_samples: int = -1,
    shuffle: bool = True,
    seed: int=0,
) -> list[dict]:
    """Load a JSON file and return a list of prompts."""
    data = load_dataset("json", data_files=file_path, split="train")
    if shuffle:
        data = data.shuffle(seed=seed)
    
    sample_count = len(data) if num_samples == -1 else min(num_samples, len(data))
    # return as list of dicts ([{"prompt": <prompt>, "tag": <tag>, directional: <directional>, fr: <fr>}] instead of {'tag': [], 'prompt': [], 'directional': [], 'fr': []})
    # or if row_keys is different from the default, return a list of dicts with the keys in row_keys
    return [
        {k: data[i][k] for k in row_keys}
        for i in range(sample_count)
    ]