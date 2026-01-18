import os
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier

try:
    # Optional import – only needed if you use --model xgb
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# -------------------------------------------------------------------
# 1. Helper: select features from your processed_data.csv
# -------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame):
    """
    Given the processed dataframe, return:
      X: features dataframe
      y: target Series ('Crime Type')
      numeric_features: list of numeric feature names
      categorical_features: list of categorical feature names
    """

    # Target column
    target_col = "Crime Type"

    # Columns to drop (IDs, raw timestamps, etc.)
    drop_cols = [
        "FIR Number",
        "Date",
        "Time",
        "DateTime",
    ]

    # Make a copy to avoid modifying original df
    data = df.copy()

    # Drop unnecessary columns if they exist
    for c in drop_cols:
        if c in data.columns:
            data = data.drop(columns=[c])

    # Separate target
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    y = data[target_col]
    X = data.drop(columns=[target_col])

    # Define numeric + categorical features explicitly
    numeric_features = []
    categorical_features = []

    # Helper: safe add if present
    def add_if_exists(col_list, col_name):
        if col_name in X.columns and col_name not in col_list:
            col_list.append(col_name)

    # Numeric
    add_if_exists(numeric_features, "Hour")
    add_if_exists(numeric_features, "Month")
    add_if_exists(numeric_features, "Year")
    add_if_exists(numeric_features, "Day_of_Week_Num")
    add_if_exists(numeric_features, "Is_Weekend")
    add_if_exists(numeric_features, "Severity Score")
    add_if_exists(numeric_features, "Latitude")
    add_if_exists(numeric_features, "Longitude")

    # Categorical
    add_if_exists(categorical_features, "Day of Week")
    add_if_exists(categorical_features, "Part of Day")
    add_if_exists(categorical_features, "Time_Category")
    add_if_exists(categorical_features, "State")
    add_if_exists(categorical_features, "District")
    add_if_exists(categorical_features, "City")
    add_if_exists(categorical_features, "Area Type")
    add_if_exists(categorical_features, "Outcome")

    # Fallback: anything remaining that's object-type → treat as categorical
    for col in X.columns:
        if col not in numeric_features and col not in categorical_features:
            if X[col].dtype == "object":
                categorical_features.append(col)

    return X, y, numeric_features, categorical_features


# -------------------------------------------------------------------
# 2. Build model (RF or XGB) with full pipeline
# -------------------------------------------------------------------
def build_pipeline(model_type: str,
                   numeric_features,
                   categorical_features):
    # Preprocessing: numeric passthrough, categorical one-hot
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    # Choose model
    model_type = model_type.lower()
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif model_type == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed. Install with `pip install xgboost`.")
        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'rf' or 'xgb'.")

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


# -------------------------------------------------------------------
# 3. Train, evaluate, and save
# -------------------------------------------------------------------
def train_and_evaluate(
    df_path: str,
    model_type: str = "rf",
    save_dir: str = "models",
):
    # 3.1 Load data
    df = pd.read_csv(df_path)

    # 3.2 Build feature matrix
    X, y_raw, numeric_features, categorical_features = build_feature_matrix(df)

    # 3.3 Encode labels (Crime Type) as integers for ML
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 3.4 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3.5 Build pipeline
    pipeline = build_pipeline(model_type, numeric_features, categorical_features)

    # 3.6 Fit model
    pipeline.fit(X_train, y_train)

    # 3.7 Evaluate
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("RESULTS")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
    print(f"F1 (weighted): {f1:.4f}\n")

    print("Classification Report:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    print("\nConfusion Matrix:\n")
    print(cm_df)

    # 3.8 Save model + label encoder
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"crime_classifier_{model_type}_{timestamp}.joblib"
    model_path = os.path.join(save_dir, model_filename)

    artifact = {
        "model": pipeline,
        "label_encoder": label_encoder,
        "feature_cols": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    joblib.dump(artifact, model_path)
    print(f"Saved model to: {model_path}")

    results = {
        "model_file": model_path,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm_df,
    }

    return results


# -------------------------------------------------------------------
# 4. CLI entrypoint
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train crime type classifier.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to processed_data.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "xgb"],
        help="Model type: 'rf' for RandomForest, 'xgb' for XGBoost.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Directory to save the trained model artifact.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train_and_evaluate(
        df_path=args.data,
        model_type=args.model,
        save_dir=args.save_dir,
    )
    print(results)
