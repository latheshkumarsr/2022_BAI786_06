
import pandas as pd
from pathlib import Path

def load_raw(path='data/raw/raw_data.csv'):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)
