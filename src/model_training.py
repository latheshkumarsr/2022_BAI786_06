import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class CrimePredictionModel:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
    
    def create_advanced_features(self, df):
        """Create advanced features for better predictions"""
        # Ensure DateTime is properly formatted
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df['DateTime'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        
        # Temporal features
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['Day_of_Year'] = df['DateTime'].dt.dayofyear
        
        # Geographic features (simplified clustering)
        df['Geo_Cluster'] = (df['Latitude'].round(1) * 10 + df['Longitude'].round(1)).astype(int)
        
        # Crime density features (rolling window)
        df['Crime_Count_7d_Rolling'] = self.calculate_rolling_crime_count(df, window=7)
        df['Severity_7d_Avg'] = self.calculate_rolling_severity(df, window=7)
        
        # Area risk score
        area_risk_scores = df.groupby('Area Type')['Severity Score'].mean().to_dict()
        df['Area_Risk_Score'] = df['Area Type'].map(area_risk_scores)
        
        return df
    
    def calculate_rolling_crime_count(self, df, window=7):
        """Calculate rolling crime count for temporal patterns"""
        # This would be implemented based on your temporal data
        return np.random.randint(1, 20, len(df))  # Placeholder
    
    def calculate_rolling_severity(self, df, window=7):
        """Calculate rolling average severity"""
        # This would be implemented based on your temporal data
        return np.random.uniform(5, 15, len(df))  # Placeholder
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Select features for modeling
        feature_columns = [
            'Latitude', 'Longitude', 'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos',
            'Day_of_Year', 'Day_of_Week_Num', 'Is_Weekend', 'Geo_Cluster',
            'Crime_Count_7d_Rolling', 'Severity_7d_Avg', 'Area_Risk_Score'
        ]
        
        # Add encoded categorical features
        categorical_cols = ['Area Type', 'Weapon Type', 'Part of Day']
        for col in categorical_cols:
            if col in df.columns:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_Encoded'] = self.encoders[col].fit_transform(df[col].fillna('Unknown'))
                feature_columns.append(f'{col}_Encoded')
        
        return df[feature_columns]
    
    def train_hotspot_model(self, df):
        """Train model to predict crime hotspots"""
        print("Training Hotspot Prediction Model...")
        
        # Create target variable (high severity = hotspot)
        df['Is_Hotspot'] = (df['Severity Score'] >= df['Severity Score'].quantile(0.75)).astype(int)
        
        # Create features
        df_with_features = self.create_advanced_features(df)
        X = self.prepare_features(df_with_features)
        y = df_with_features['Is_Hotspot']
        
        # Remove any rows with NaN values
        valid_indices = ~X.isna().any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Hotspot Model Accuracy: {accuracy:.4f}")
        
        # Store feature importance
        self.feature_importance['hotspot'] = dict(zip(X.columns, model.feature_importances_))
        
        self.models['hotspot'] = model
        return model, accuracy
    
    def train_crime_type_model(self, df):
        """Train model to predict crime type"""
        print("Training Crime Type Prediction Model...")
        
        # Filter to most common crime types for better accuracy
        common_crimes = df['Crime Type'].value_counts().head(8).index
        df_filtered = df[df['Crime Type'].isin(common_crimes)].copy()
        
        # Create features
        df_with_features = self.create_advanced_features(df_filtered)
        X = self.prepare_features(df_with_features)
        y = df_filtered['Crime Type']
        
        # Remove any rows with NaN values
        valid_indices = ~X.isna().any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Encode target
        self.encoders['crime_type'] = LabelEncoder()
        y_encoded = self.encoders['crime_type'].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Crime Type Model Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.encoders['crime_type'].classes_))
        
        self.models['crime_type'] = model
        return model, accuracy
    
    def train_weapon_prediction_model(self, df):
        """Train model to predict weapon type"""
        print("Training Weapon Prediction Model...")
        
        # Filter to common weapons
        common_weapons = df['Weapon Type'].value_counts().head(6).index
        df_filtered = df[df['Weapon Type'].isin(common_weapons)].copy()
        
        # Create features
        df_with_features = self.create_advanced_features(df_filtered)
        X = self.prepare_features(df_with_features)
        y = df_filtered['Weapon Type']
        
        # Remove any rows with NaN values
        valid_indices = ~X.isna().any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Encode target
        self.encoders['weapon_type'] = LabelEncoder()
        y_encoded = self.encoders['weapon_type'].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Weapon Prediction Model Accuracy: {accuracy:.4f}")
        
        self.models['weapon'] = model
        return model, accuracy
    
    def train_severity_prediction_model(self, df):
        """Train model to predict severity score"""
        print("Training Severity Prediction Model...")
        
        # Create features
        df_with_features = self.create_advanced_features(df)
        X = self.prepare_features(df_with_features)
        y = df['Severity Score']
        
        # Remove any rows with NaN values
        valid_indices = ~X.isna().any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = np.mean(np.abs(y_pred - y_test))
        print(f"Severity Prediction MAE: {mae:.4f}")
        
        self.models['severity'] = model
        return model, mae
    
    def save_models(self, path="models/"):
        """Save all trained models and encoders"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}_model.pkl")
            print(f"Saved {name} model")
        
        for name, encoder in self.encoders.items():
            joblib.dump(encoder, f"{path}/{name}_encoder.pkl")
            print(f"Saved {name} encoder")
        
        # Save feature importance
        joblib.dump(self.feature_importance, f"{path}/feature_importance.pkl")
        
        print("All models saved successfully!")
    
    def load_models(self, path="models/"):
        """Load pre-trained models and encoders"""
        try:
            model_files = {
                'hotspot': 'hotspot_model.pkl',
                'crime_type': 'crime_type_model.pkl', 
                'weapon': 'weapon_model.pkl',
                'severity': 'severity_model.pkl'
            }
            
            for name, filename in model_files.items():
                self.models[name] = joblib.load(f"{path}/{filename}")
                print(f"Loaded {name} model")
            
            # Load encoders
            encoder_files = [f for f in os.listdir(path) if 'encoder' in f]
            for filename in encoder_files:
                name = filename.replace('_encoder.pkl', '')
                self.encoders[name] = joblib.load(f"{path}/{filename}")
                print(f"Loaded {name} encoder")
            
            print("All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Usage example
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data/processed_data.csv')
    
    # Initialize and train models
    predictor = CrimePredictionModel()
    
    # Train all models
    hotspot_model, hotspot_acc = predictor.train_hotspot_model(df)
    crime_type_model, crime_type_acc = predictor.train_crime_type_model(df)
    weapon_model, weapon_acc = predictor.train_weapon_prediction_model(df)
    severity_model, severity_mae = predictor.train_severity_prediction_model(df)
    
    # Save models
    predictor.save_models()