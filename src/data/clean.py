
import pandas as pd

def downcast_df(df):
    for col in df.select_dtypes(include=['int','float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='unsigned')
    return df
