import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CrimeDataProcessor:
    def __init__(self, data_path='data/processed_data.csv'):
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and process the entire dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded dataset with {len(self.df)} records")
            
            # Convert date columns
            date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                self.df['DateTime'] = pd.to_datetime(self.df[date_columns[0]], errors='coerce')
            
            # Ensure numeric columns
            self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
            self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
            self.df['Severity Score'] = pd.to_numeric(self.df['Severity Score'], errors='coerce')
            
            # Remove rows with missing critical data
            self.df = self.df.dropna(subset=['Latitude', 'Longitude', 'Severity Score'])
            print(f"✅ Cleaned dataset: {len(self.df)} valid records")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def calculate_real_time_features(self):
        """Calculate real features from the dataset"""
        print("🔄 Calculating real-time features from dataset...")
        
        # Group by location and time to create realistic features
        location_features = self.calculate_location_based_features()
        temporal_features = self.calculate_temporal_features()
        crime_patterns = self.analyze_crime_patterns()
        
        return {
            'location_features': location_features,
            'temporal_features': temporal_features,
            'crime_patterns': crime_patterns
        }
    
    def calculate_location_based_features(self):
        """Calculate actual crime density by location"""
        # Group by rounded coordinates to create clusters
        self.df['Lat_Rounded'] = self.df['Latitude'].round(2)
        self.df['Lon_Rounded'] = self.df['Longitude'].round(2)
        
        location_stats = self.df.groupby(['Lat_Rounded', 'Lon_Rounded']).agg({
            'FIR Number': 'count',
            'Severity Score': ['mean', 'max'],
            'Crime Type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        
        location_stats.columns = ['Latitude', 'Longitude', 'Crime_Count', 'Avg_Severity', 'Max_Severity', 'Most_Common_Crime']
        
        # Calculate crime density (crimes per square km approximation)
        location_stats['Crime_Density'] = location_stats['Crime_Count'] / location_stats['Crime_Count'].max()
        
        return location_stats
    
    def calculate_temporal_features(self):
        """Calculate actual temporal patterns"""
        temporal_stats = {}
        
        # Crimes by hour
        temporal_stats['hourly_pattern'] = self.df['Hour'].value_counts().sort_index()
        
        # Crimes by day of week
        if 'Day_of_Week_Num' in self.df.columns:
            temporal_stats['daily_pattern'] = self.df['Day_of_Week_Num'].value_counts().sort_index()
        
        # Crimes by month
        temporal_stats['monthly_pattern'] = self.df['Month'].value_counts().sort_index()
        
        # Crimes by area type
        temporal_stats['area_pattern'] = self.df['Area Type'].value_counts()
        
        return temporal_stats
    
    def analyze_crime_patterns(self):
        """Analyze actual crime patterns from data"""
        patterns = {}
        
        # Crime type distribution
        patterns['crime_type_dist'] = self.df['Crime Type'].value_counts()
        
        # Weapon usage patterns
        patterns['weapon_dist'] = self.df['Weapon Type'].value_counts()
        
        # Crime outcomes
        patterns['outcome_dist'] = self.df['Outcome'].value_counts()
        
        # Area type risk analysis
        area_risk = self.df.groupby('Area Type').agg({
            'Severity Score': 'mean',
            'FIR Number': 'count'
        }).round(2)
        patterns['area_risk'] = area_risk
        
        return patterns
    
    def get_crimes_near_location(self, latitude, longitude, radius_km=5):
        """Get actual crimes near a specific location"""
        # Simplified distance calculation (approximate)
        lat_diff = (self.df['Latitude'] - latitude).abs()
        lon_diff = (self.df['Longitude'] - longitude).abs()
        
        # Rough approximation: 1 degree ≈ 111 km
        distance_km = np.sqrt((lat_diff * 111)**2 + (lon_diff * 111)**2)
        
        nearby_crimes = self.df[distance_km <= radius_km]
        return nearby_crimes
    
    def get_temporal_risk(self, hour, day_of_week, month):
        """Calculate actual temporal risk from historical data"""
        # Filter data for similar temporal conditions
        temporal_match = self.df[
            (self.df['Hour'] == hour) & 
            (self.df['Day_of_Week_Num'] == day_of_week) & 
            (self.df['Month'] == month)
        ]
        
        if len(temporal_match) > 0:
            risk_score = temporal_match['Severity Score'].mean() / 20  # Normalize to 0-1
            crime_count = len(temporal_match)
        else:
            # Fallback to overall averages
            risk_score = self.df['Severity Score'].mean() / 20
            crime_count = len(self.df) / (24 * 7 * 12)  # Average per hour-day-month
        
        return min(1.0, risk_score), crime_count
    
    def get_location_risk(self, latitude, longitude):
        """Calculate actual location risk from historical data"""
        nearby_crimes = self.get_crimes_near_location(latitude, longitude, radius_km=2)
        
        if len(nearby_crimes) > 0:
            location_risk = nearby_crimes['Severity Score'].mean() / 20
            crime_density = len(nearby_crimes) / len(self.df)
        else:
            # Use overall averages for new locations
            location_risk = self.df['Severity Score'].mean() / 20
            crime_density = 0.01  # Default low density
        
        return min(1.0, location_risk), crime_density, nearby_crimes
    
    def get_area_type_risk(self, area_type):
        """Calculate actual risk for specific area type"""
        area_crimes = self.df[self.df['Area Type'] == area_type]
        
        if len(area_crimes) > 0:
            area_risk = area_crimes['Severity Score'].mean() / 20
            area_frequency = len(area_crimes) / len(self.df)
        else:
            area_risk = self.df['Severity Score'].mean() / 20
            area_frequency = 0.1
        
        return min(1.0, area_risk), area_frequency
    
    def predict_crime_types_for_location(self, latitude, longitude, hour, area_type):
        """Predict crime types based on actual historical patterns"""
        # Get crimes in similar conditions
        similar_crimes = self.df[
            (self.df['Latitude'].between(latitude-0.1, latitude+0.1)) &
            (self.df['Longitude'].between(longitude-0.1, longitude+0.1)) &
            (self.df['Hour'].between(hour-2, hour+2)) &
            (self.df['Area Type'] == area_type)
        ]
        
        if len(similar_crimes) == 0:
            # Broaden search if no exact matches
            similar_crimes = self.df[
                (self.df['Area Type'] == area_type) &
                (self.df['Hour'].between(hour-3, hour+3))
            ]
        
        if len(similar_crimes) > 0:
            crime_distribution = similar_crimes['Crime Type'].value_counts(normalize=True)
            results = []
            
            for crime_type, probability in crime_distribution.head(5).items():
                results.append({
                    'crime_type': crime_type,
                    'probability': probability,
                    'confidence': 'High' if probability > 0.3 else 'Medium'
                })
            
            return results
        else:
            # Return overall distribution
            overall_dist = self.df['Crime Type'].value_counts(normalize=True).head(3)
            return [{'crime_type': ct, 'probability': prob, 'confidence': 'Low'} 
                   for ct, prob in overall_dist.items()]
    
    def predict_weapons_for_location(self, latitude, longitude, crime_type):
        """Predict weapon usage based on actual patterns"""
        # Get weapons used for similar crimes in nearby locations
        similar_cases = self.df[
            (self.df['Latitude'].between(latitude-0.2, latitude+0.2)) &
            (self.df['Longitude'].between(longitude-0.2, longitude+0.2)) &
            (self.df['Crime Type'] == crime_type)
        ]
        
        if len(similar_cases) == 0:
            # Use overall weapon distribution for this crime type
            similar_cases = self.df[self.df['Crime Type'] == crime_type]
        
        if len(similar_cases) > 0:
            weapon_dist = similar_cases['Weapon Type'].value_counts(normalize=True)
            results = []
            
            for weapon, probability in weapon_dist.head(4).items():
                results.append({
                    'weapon_type': weapon,
                    'probability': probability
                })
            
            return results
        else:
            # Return overall weapon distribution
            overall_weapons = self.df['Weapon Type'].value_counts(normalize=True).head(3)
            return [{'weapon_type': wt, 'probability': prob} 
                   for wt, prob in overall_weapons.items()]

# Example usage
if __name__ == "__main__":
    processor = CrimeDataProcessor()
    features = processor.calculate_real_time_features()
    print(f"📍 Location features: {len(features['location_features'])} clusters")
    print(f"🕒 Temporal patterns: {len(features['temporal_features'])} categories")
    print(f"📊 Crime patterns: {len(features['crime_patterns'])} analyses")