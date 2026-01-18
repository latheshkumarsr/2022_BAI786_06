
from pathlib import Path
import pandas as pd

def save_parquet(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
