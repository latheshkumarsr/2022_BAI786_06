
import pandas as pd

def add_time_features(df, date_col='DATE OCC'):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['hour'] = df[date_col].dt.hour
    return df
