
# helpers to load model and predict
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parents[2] / 'models' / 'crime_pipeline.joblib'

def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def predict_from_model(model, features):
    if model is None:
        return None
    return model.predict_proba([features]) if hasattr(model, 'predict_proba') else model.predict([features])
