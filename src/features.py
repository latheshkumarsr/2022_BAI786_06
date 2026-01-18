# src/features.py
import pandas as pd
import numpy as np
import os

def load_data(path='data/processed_data.csv'):
    """Load CSV and do minimal cleaning. Returns pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    # try to parse a date/time column
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        df['DateTime'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    return df

def prepare_ml_dataframe(df, target_col='Crime Type', drop_na=True):
    """
    Build a cleaned ML-ready DataFrame and a list of feature column names.
    Adjust the candidate features to match your actual dataset's columns.
    """
    candidate_features = []
    # conservative defaults - edit if your dataset uses different column names
    for c in ['Latitude','Longitude','Hour','Month','Year','Severity Score',
              'Area Type','Weapon Type','State','City']:
        if c in df.columns and c != target_col:
            candidate_features.append(c)

    # create ml_df including the target
    ml_df = df[candidate_features + [target_col]].copy()
    ml_df = ml_df.rename(columns={target_col: 'target'})

    if drop_na:
        ml_df = ml_df.dropna(subset=['target'])

    # numeric / categorical treatment: fill missing
    for col in candidate_features:
        if col in ml_df.columns:
            if ml_df[col].dtype.kind in 'biufc':  # numeric
                ml_df[col] = ml_df[col].fillna(ml_df[col].median())
            else:
                ml_df[col] = ml_df[col].fillna('Unknown').astype(str)

    # ensure target is string
    ml_df['target'] = ml_df['target'].astype(str)
    return ml_df, candidate_features
