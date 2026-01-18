import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import sys
import math

# Add parent directory to path to import data_processor
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import CrimeDataProcessor from project, if not available create a lightweight fallback
try:
    from src.data_processor import CrimeDataProcessor  # type: ignore
    _HAS_PROCESSOR = True
except Exception:
    _HAS_PROCESSOR = False


# ----- Helper utilities -----

def haversine(lat1, lon1, lat2, lon2):
    """Return distance in kilometers between two lat/lon points."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class _FallbackCrimeDataProcessor:
    """A lightweight fallback data processor that works directly from a CSV when
    the project's src.data_processor is not importable. It implements the minimal
    interface expected by DatabaseDrivenPredictionEngine.
    """

    def __init__(self, csv_path: str = r"D:\crime_pattern_prediction\data\processed\dataset.csv"):
        self.csv_path = csv_path
        self.df = self._load_csv(csv_path)

        # Ensure useful column names exist or create reasonable defaults
        self._standardize_columns()

    def _load_csv(self, path):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            st.error(f"Fallback processor failed to read CSV at {path}: {e}")
            df = pd.DataFrame()
        return df

    def _standardize_columns(self):
        if self.df.empty:
            return
        # Trim column names
        self.df.columns = [c.strip() for c in self.df.columns]

        # common mappings
        colmap = {
            'LAT': 'Latitude', 'Lat': 'Latitude', 'lat': 'Latitude',
            'LON': 'Longitude', 'Lon': 'Longitude', 'lon': 'Longitude',
            'Crm Cd Desc': 'Crime Type', 'Crm Cd': 'Crm Code',
            'DATE OCC': 'DateTime', 'Date Rptd': 'DateReported', 'TIME OCC': 'TimeOccurred',
            'occ_year': 'occ_year', 'occ_month': 'occ_month', 'occ_day': 'occ_day',
            'STATUS DESC': 'Status Desc'
        }
        for a, b in colmap.items():
            if a in self.df.columns and b not in self.df.columns:
                self.df = self.df.rename(columns={a: b})

        # Ensure lat/lon numeric
        for c in ('Latitude', 'Longitude'):
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

        # Date/time
        if 'DateTime' in self.df.columns:
            self.df['DateTime'] = pd.to_datetime(self.df['DateTime'], errors='coerce')
        else:
            # try other date columns
            for cand in ['DateReported', 'DATE OCC', 'Date Occurred', 'DATE_OCC']:
                if cand in self.df.columns:
                    self.df['DateTime'] = pd.to_datetime(self.df[cand], errors='coerce')
                    break

        # Hour
        if 'Hour' not in self.df.columns:
            if 'TimeOccurred' in self.df.columns:
                def parse_hour(v):
                    try:
                        if pd.isna(v):
                            return np.nan
                        s = str(v)
                        if ':' in s:
                            return pd.to_datetime(s).hour
                        if s.isdigit():
                            s = s.zfill(4)
                            return int(s[:2])
                        return np.nan
                    except Exception:
                        return np.nan
                self.df['Hour'] = self.df['TimeOccurred'].apply(parse_hour)
            elif 'DateTime' in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df['DateTime']):
                self.df['Hour'] = self.df['DateTime'].dt.hour

        # Crime Type
        if 'Crime Type' not in self.df.columns:
            # fallback to any descriptive column
            for c in ['Crm Cd Desc', 'REMARKS', 'Crm Cd']:
                if c in self.df.columns:
                    self.df['Crime Type'] = self.df[c].astype(str)
                    break

        # Severity Score
        if 'Severity Score' not in self.df.columns:
            # infer from crime type keywords
            def infer_sev(x):
                if pd.isna(x):
                    return 1
                s = str(x).lower()
                if any(k in s for k in ['murder', 'homicide']):
                    return 10
                if any(k in s for k in ['assault']):
                    return 7
                if any(k in s for k in ['robbery']):
                    return 8
                if any(k in s for k in ['burglary']):
                    return 6
                if any(k in s for k in ['theft', 'vehicle']):
                    return 4
                return 3
            self.df['Severity Score'] = self.df.get('Crime Type', '').apply(infer_sev)

        # Weapon Type
        if 'Weapon Type' not in self.df.columns:
            self.df['Weapon Type'] = 'Unknown'

        # State and Area Type defaults
        if 'State' not in self.df.columns:
            self.df['State'] = self.df.get('AREA NAME', 'Unknown')
        if 'Area Type' not in self.df.columns:
            self.df['Area Type'] = 'Unknown'

        # Drop rows without coords
        self.df = self.df.dropna(subset=['Latitude', 'Longitude']) if 'Latitude' in self.df.columns and 'Longitude' in self.df.columns else self.df

    # Interface methods expected by engine
    def calculate_real_time_features(self):
        # create simple location groups by rounding lat/lon
        if self.df.empty:
            return {'location_features': []}
        df = self.df.copy()
        df['lat_r'] = df['Latitude'].round(3)
        df['lon_r'] = df['Longitude'].round(3)
        loc_groups = df.groupby(['lat_r', 'lon_r']).size().reset_index(name='count')
        location_features = loc_groups.to_dict('records')
        return {'location_features': location_features}

    def get_temporal_risk(self, hour, day_of_week, month):
        # compute normalized frequency for hour, day, month
        if self.df.empty:
            return 0.1, 0
        df = self.df
        # Hour risk
        if 'Hour' in df.columns:
            hour_counts = df['Hour'].value_counts()
            hcount = int(hour_counts.get(hour, 0))
            hour_risk = hcount / max(1, hour_counts.max())
        else:
            hour_risk = 0.1
        # Day risk
        if 'occ_day' in df.columns:
            day_counts = df['occ_day'].value_counts()
            dcount = int(day_counts.get(day_of_week, 0))
            day_risk = dcount / max(1, day_counts.max())
        else:
            day_risk = 0.1
        # Month risk
        if 'occ_month' in df.columns:
            mcounts = df['occ_month'].value_counts()
            mcount = int(mcounts.get(month, 0))
            month_risk = mcount / max(1, mcounts.max())
        else:
            month_risk = 0.1
        # combine
        temporal_risk = (hour_risk * 0.5 + day_risk * 0.3 + month_risk * 0.2)
        temporal_count = len(df[(df['Hour'] == hour) & (df.get('occ_day') == day_of_week)]) if 'Hour' in df.columns else 0
        return float(temporal_risk), int(temporal_count)

    def get_location_risk(self, latitude, longitude, radius_km=1.0):
        if self.df.empty:
            return 0.1, 0, pd.DataFrame()
        df = self.df
        # compute distances
        distances = df.apply(lambda r: haversine(latitude, longitude, r['Latitude'], r['Longitude']), axis=1)
        nearby = df[distances <= radius_km].copy()
        density = len(nearby)
        max_density = df.shape[0] if df.shape[0] > 0 else 1
        location_risk = min(1.0, density / max(1, df.shape[0] / 50))  # heuristic
        return float(location_risk), int(density), nearby

    def get_area_type_risk(self, area_type):
        if self.df.empty:
            return 0.1, 0
        area_counts = self.df['Area Type'].value_counts()
        a_count = int(area_counts.get(area_type, 0))
        area_risk = a_count / max(1, area_counts.max())
        return float(area_risk), int(a_count)

    def predict_crime_types_for_location(self, latitude, longitude, hour, area_type, radius_km=1.0, top_k=5):
        if self.df.empty:
            return []
        df = self.df
        distances = df.apply(lambda r: haversine(latitude, longitude, r['Latitude'], r['Longitude']), axis=1)
        nearby = df[distances <= radius_km].copy()
        if hour is not None and 'Hour' in nearby.columns:
            nearby = nearby[nearby['Hour'].fillna(-1) == hour]
        if nearby.empty:
            # fallback to area_type
            nearby = df[df['Area Type'] == area_type]
        top = nearby['Crime Type'].value_counts().head(top_k)
        total = top.sum() if top.sum() > 0 else 1
        results = [{'crime_type': ct, 'probability': float(cnt / total)} for ct, cnt in top.items()]
        return results

    def predict_weapons_for_location(self, latitude, longitude, crime_type=None, radius_km=1.0, top_k=5):
        if self.df.empty:
            return []
        df = self.df
        distances = df.apply(lambda r: haversine(latitude, longitude, r['Latitude'], r['Longitude']), axis=1)
        nearby = df[distances <= radius_km].copy()
        if crime_type:
            nearby = nearby[nearby['Crime Type'] == crime_type]
        if nearby.empty:
            nearby = df
        top = nearby['Weapon Type'].value_counts().head(top_k)
        total = top.sum() if top.sum() > 0 else 1
        return [{'weapon_type': w, 'probability': float(c / total)} for w, c in top.items()]

    def get_crimes_near_location(self, latitude, longitude, radius_km=3.0):
        if self.df.empty:
            return pd.DataFrame()
        distances = self.df.apply(lambda r: haversine(latitude, longitude, r['Latitude'], r['Longitude']), axis=1)
        return self.df[distances <= radius_km].copy()


# Choose processor implementation
ProcessorImpl = CrimeDataProcessor if _HAS_PROCESSOR else _FallbackCrimeDataProcessor


class DatabaseDrivenPredictionEngine:
    def __init__(self, csv_path: str | None = None):
        self.processor = None
        self.dataset_stats = {}
        self.loaded = False
        self.csv_path = csv_path or r"D:\crime_pattern_prediction\data\processed\dataset.csv"
        self.initialize_processor()

    def initialize_processor(self):
        """Initialize the data processor with actual dataset"""
        try:
            if _HAS_PROCESSOR:
                # use the project's processor which might accept a path or load internally
                self.processor = ProcessorImpl()
            else:
                self.processor = ProcessorImpl(self.csv_path)

            if getattr(self.processor, 'df', None) is not None and not self.processor.df.empty:
                self.dataset_stats = self.processor.calculate_real_time_features()
                self.loaded = True
                st.success(f"✅ Loaded real dataset with {len(self.processor.df)} crime records")
                st.success(f"📍 Analyzing {len(self.dataset_stats.get('location_features', []))} crime clusters")
            else:
                st.error("❌ Could not load crime dataset or dataset is empty")
        except Exception as e:
            st.error(f"❌ Error initializing data processor: {e}")

    def predict_hotspot_risk(self, input_data):
        """Predict hotspot risk using actual dataset patterns"""
        if not self.loaded:
            return {"error": "Dataset not loaded"}

        try:
            latitude = float(input_data.get('latitude'))
            longitude = float(input_data.get('longitude'))
            hour = input_data.get('hour')
            day_of_week = input_data.get('day_of_week')
            month = input_data.get('month')
            area_type = input_data.get('area_type', 'Unknown')

            temporal_risk, temporal_count = self.processor.get_temporal_risk(hour, day_of_week, month)
            location_risk, location_density, nearby_crimes = self.processor.get_location_risk(latitude, longitude)
            area_risk, area_frequency = self.processor.get_area_type_risk(area_type)

            total_risk = float(temporal_risk * 0.3 + location_risk * 0.4 + area_risk * 0.3)

            total_crimes_nearby = int(len(nearby_crimes)) if hasattr(nearby_crimes, '__len__') else int(nearby_crimes)
            if total_crimes_nearby > 10:
                total_risk = min(1.0, total_risk * 1.2)
            elif total_crimes_nearby == 0:
                total_risk = max(0.05, total_risk * 0.7)

            risk_level = "High" if total_risk > 0.7 else "Medium" if total_risk > 0.4 else "Low"

            return {
                'risk_score': round(total_risk, 3),
                'risk_level': risk_level,
                'confidence': round(min(0.95, 0.6 + (total_crimes_nearby / 150)), 3),
                'nearby_crimes_count': total_crimes_nearby,
                'temporal_risk': round(float(temporal_risk), 3),
                'location_risk': round(float(location_risk), 3),
                'area_risk': round(float(area_risk), 3)
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_crime_type(self, input_data):
        """Predict crime types using actual dataset patterns"""
        if not self.loaded:
            return {"error": "Dataset not loaded"}

        try:
            latitude = float(input_data.get('latitude'))
            longitude = float(input_data.get('longitude'))
            hour = input_data.get('hour')
            area_type = input_data.get('area_type', 'Unknown')

            predictions = self.processor.predict_crime_types_for_location(latitude, longitude, hour, area_type)
            return predictions

        except Exception as e:
            return {"error": str(e)}

    def predict_weapon_usage(self, input_data, predicted_crime_type=None):
        """Predict weapon usage using actual dataset patterns"""
        if not self.loaded:
            return {"error": "Dataset not loaded"}

        try:
            latitude = float(input_data.get('latitude'))
            longitude = float(input_data.get('longitude'))

            if predicted_crime_type is None:
                crime_predictions = self.predict_crime_type(input_data)
                if crime_predictions and isinstance(crime_predictions, list) and len(crime_predictions) > 0:
                    predicted_crime_type = crime_predictions[0].get('crime_type')
                else:
                    predicted_crime_type = 'Theft'

            weapons = self.processor.predict_weapons_for_location(latitude, longitude, predicted_crime_type)
            return weapons

        except Exception as e:
            return {"error": str(e)}

    def predict_severity(self, input_data):
        """Predict severity using actual dataset patterns"""
        if not self.loaded:
            return {"error": "Dataset not loaded"}

        try:
            latitude = float(input_data.get('latitude'))
            longitude = float(input_data.get('longitude'))
            area_type = input_data.get('area_type', 'Unknown')

            nearby_crimes = self.processor.get_crimes_near_location(latitude, longitude, radius_km=3)

            if len(nearby_crimes) > 0:
                avg_severity = nearby_crimes['Severity Score'].mean()
                max_severity = nearby_crimes['Severity Score'].max()
                predicted_severity = float(avg_severity * 0.7 + max_severity * 0.3)
            else:
                area_crimes = self.processor.df[self.processor.df.get('Area Type') == area_type] if 'Area Type' in self.processor.df.columns else pd.DataFrame()
                predicted_severity = float(area_crimes['Severity Score'].mean()) if len(area_crimes) > 0 else 8.0

            predicted_severity = max(1.0, min(20.0, predicted_severity))
            severity_level = 'High' if predicted_severity > 15 else 'Medium' if predicted_severity > 8 else 'Low'

            return {
                'severity_score': round(predicted_severity, 1),
                'severity_level': severity_level,
                'based_on_crimes': len(nearby_crimes)
            }

        except Exception as e:
            return {"error": str(e)}

    def generate_comprehensive_prediction(self, input_data):
        """Generate comprehensive prediction using actual dataset"""
        results = {
            'hotspot_risk': self.predict_hotspot_risk(input_data),
            'crime_types': self.predict_crime_type(input_data),
            'severity': self.predict_severity(input_data),
            'timestamp': datetime.now(),
            'dataset_info': {
                'total_crimes': len(self.processor.df) if getattr(self.processor, 'df', None) is not None else 0,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
        }

        if isinstance(results['crime_types'], list) and results['crime_types']:
            crime_type = results['crime_types'][0].get('crime_type')
            results['weapons'] = self.predict_weapon_usage(input_data, crime_type)
        else:
            results['weapons'] = self.predict_weapon_usage(input_data)

        if 'error' not in results['hotspot_risk']:
            risk_score = results['hotspot_risk'].get('risk_score', 0)
            safety_score = max(0, 100 - (risk_score * 100))
            results['safety_score'] = int(round(safety_score))

        return results

    def get_dataset_insights(self):
        """Get insights about the dataset"""
        if not self.loaded:
            return {}

        df = self.processor.df
        # choose a datetime column if available
        dt_col = 'DateTime' if 'DateTime' in df.columns else None
        if dt_col and pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            start = df[dt_col].min().strftime('%Y-%m-%d')
            end = df[dt_col].max().strftime('%Y-%m-%d')
        else:
            start = None
            end = None

        insights = {
            'total_records': len(df),
            'date_range': {'start': start, 'end': end},
            'top_crimes': df['Crime Type'].value_counts().head(5).to_dict() if 'Crime Type' in df.columns else {},
            'top_states': df['State'].value_counts().head(5).to_dict() if 'State' in df.columns else {},
            'severity_stats': {
                'average': float(df['Severity Score'].mean()) if 'Severity Score' in df.columns else None,
                'max': float(df['Severity Score'].max()) if 'Severity Score' in df.columns else None,
                'min': float(df['Severity Score'].min()) if 'Severity Score' in df.columns else None,
            },
            'weapon_stats': df['Weapon Type'].value_counts().head(5).to_dict() if 'Weapon Type' in df.columns else {}
        }
        return insights


# Visualization & recommendations functions are intentionally left almost unchanged
# (they were well formed). We simply keep them and ensure they work with our outputs.

def create_prediction_visualizations(prediction_results, dataset_insights=None):
    visualizations = {}
    if 'safety_score' in prediction_results:
        safety_score = prediction_results['safety_score']
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = safety_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Safety Score\n(Based on {prediction_results.get('dataset_info', {}).get('total_crimes', 0)} crimes)", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': 'red'},
                    {'range': [40, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'green'}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': safety_score}
            }
        ))
        fig_gauge.update_layout(height=300)
        visualizations['safety_gauge'] = fig_gauge

    if 'crime_types' in prediction_results and isinstance(prediction_results['crime_types'], list):
        crime_data = prediction_results['crime_types']
        if crime_data:
            crimes = [item['crime_type'] for item in crime_data]
            probs = [item['probability'] * 100 for item in crime_data]
            fig_crimes = px.bar(x=probs, y=crimes, orientation='h', title="Predicted Crime Types (Based on Historical Patterns)", labels={'x': 'Probability (%)', 'y': 'Crime Type'})
            fig_crimes.update_layout(height=300)
            visualizations['crime_probabilities'] = fig_crimes

    if 'weapons' in prediction_results and isinstance(prediction_results['weapons'], list):
        weapon_data = prediction_results['weapons']
        if weapon_data:
            weapons = [item['weapon_type'] for item in weapon_data]
            probs = [item['probability'] * 100 for item in weapon_data]
            fig_weapons = px.pie(values=probs, names=weapons, title="Predicted Weapon Usage (Historical Patterns)")
            fig_weapons.update_layout(height=300)
            visualizations['weapon_probabilities'] = fig_weapons

    if dataset_insights:
        crimes = list(dataset_insights.get('top_crimes', {}).keys())[:5]
        counts = list(dataset_insights.get('top_crimes', {}).values())[:5]
        if crimes and counts:
            fig_dataset = px.bar(x=counts, y=crimes, orientation='h', title="Most Common Crimes in Dataset")
            fig_dataset.update_layout(height=250)
            visualizations['dataset_overview'] = fig_dataset

    return visualizations


def generate_safety_recommendations(results, input_data):
    recommendations = []
    hour = input_data.get('hour')
    try:
        if hour is not None and (int(hour) >= 20 or int(hour) <= 6):
            recommendations.append("🌙 High crime activity historically during night hours - avoid traveling alone")
            recommendations.append("💡 Historical data shows increased risks in poorly lit areas after dark")
    except Exception:
        pass

    if 'hotspot_risk' in results and isinstance(results['hotspot_risk'], dict) and 'risk_level' in results['hotspot_risk']:
        risk_level = results['hotspot_risk']['risk_level']
        nearby_count = results['hotspot_risk'].get('nearby_crimes_count', 0)
        if risk_level == "High":
            recommendations.append(f"🚨 Historical data shows {nearby_count} crimes nearby - high-risk area detected")
            recommendations.append("📱 Emergency services response analysis: keep contacts accessible")
        elif risk_level == "Medium":
            recommendations.append(f"⚠️ Moderate risk area with {nearby_count} historical incidents - remain alert")

    if 'crime_types' in results and isinstance(results['crime_types'], list) and results['crime_types']:
        top_crime = results['crime_types'][0]['crime_type']
        top_prob = results['crime_types'][0]['probability'] * 100
        if top_crime in ['Vehicle Theft', 'Robbery']:
            recommendations.append(f"🔒 {top_prob:.1f}% probability of {top_crime} - secure vehicles")
        elif top_crime in ['Assault', 'Murder']:
            recommendations.append(f"🚶 {top_prob:.1f}% probability of violent crime - avoid isolated areas")

    area_type = input_data.get('area_type')
    if area_type == 'Industrial Area':
        recommendations.append("🏭 Historical data shows reduced visibility crimes in industrial areas")
    elif area_type == 'Slum Area':
        recommendations.append("🏚️ Crime density analysis shows higher incidence rates in this area type")
    elif area_type == 'Park Area':
        recommendations.append("🌳 Temporal analysis shows increased risks during evening hours in parks")

    recommendations.extend([
        "✅ Crime pattern analysis: always be aware of your surroundings",
        "✅ Historical response data: trust your instincts and leave if uncomfortable",
        "✅ Dataset analysis: keep communication devices charged and accessible"
    ])

    return recommendations[:8]
