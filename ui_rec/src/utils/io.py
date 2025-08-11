import os
import glob
from datetime import datetime
import yaml
import pandas as pd
from joblib import dump, load


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_csv_with_timestamp(df: pd.DataFrame, base_name: str, out_dir: str) -> str:
    ensure_dir(out_dir)
    ts = timestamp()
    path = os.path.join(out_dir, f"{base_name}_{ts}.csv")
    df.to_csv(path, index=False)
    return path


def latest_file(pattern: str, directory: str) -> str:
    candidates = sorted(glob.glob(os.path.join(directory, pattern)))
    return candidates[-1] if candidates else ""


def save_model(model, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    dump(model, out_path)


def load_model(path: str):
    return load(path)


def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) 