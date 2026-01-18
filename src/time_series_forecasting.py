import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib

def train_time_series_model():
    """Train a time series forecasting model for crime prediction."""
    
    # Load prepared time series data
    df = pd.read_csv(Path(__file__).parent.parent / 'data' / 'processed' / 'daily_crime_data.csv', index_col='Date', parse_dates=True)
    
    # Create lag features for time series
    for lag in range(1, 8):  # 7 days of lag
        df[f'lag_{lag}'] = df['Crime_Count'].shift(lag)
    
    # Create rolling features
    df['rolling_mean_7'] = df['Crime_Count'].rolling(window=7).mean()
    df['rolling_std_7'] = df['Crime_Count'].rolling(window=7).std()
    
    # Drop NaN values
    df = df.dropna()
    
    # Prepare features and target
    features = [col for col in df.columns if col != 'Crime_Count']
    X = df[features]
    y = df['Crime_Count']
    
    # Split data (last 30 days for testing)
    split_index = -30
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Time Series Model MAE: {mae:.2f}")
    print(f"Average daily crimes: {y.mean():.2f}")
    
    # Save model
    joblib.dump(model, Path(__file__).parent.parent / 'models' / 'time_series_model.pkl')
    print("✅ Time series model trained and saved!")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', marker='o')
    plt.plot(y_test.index, y_pred, label='Predicted', marker='x')
    plt.title('Crime Time Series Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Number of Crimes')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'reports' / 'figures' / 'time_series_forecast.png', dpi=300)
    plt.close()
    
    return model, mae

if __name__ == "__main__":
    print("Training time series forecasting model...")
    model, mae = train_time_series_model()
    print(f"Model trained with Mean Absolute Error: {mae:.2f}")