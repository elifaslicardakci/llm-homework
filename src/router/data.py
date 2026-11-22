from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def load_split(name: str) -> pd.DataFrame:
    """name: 'train', 'val' or 'test'. Columns: text, intent, domain, team, lang, source."""
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


def label_list() -> list[str]:
    """Fixed, sorted list of all intents (including 'oos'); the order of model outputs."""
    return sorted(load_split("train")["intent"].unique())
