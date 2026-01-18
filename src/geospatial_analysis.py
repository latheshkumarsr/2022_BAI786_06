import pandas as pd
import folium
from folium.plugins import HeatMap
from pathlib import Path
import json

# Define file paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
DATA_PROCESSED_PATH = project_root / 'data' / 'processed'
APP_ASSETS_PATH = project_root / 'app' / 'assets'

def create_minimal_geojson():
    """Creates a minimal GeoJSON for major Indian states with approximate coordinates."""
    india_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"st_nm": "Maharashtra"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[72.0, 19.0], [72.0, 21.0], [80.0, 21.0], [80.0, 19.0], [72.0, 19.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"st_nm": "Delhi"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[76.5, 28.0], [76.5, 29.0], [77.5, 29.0], [77.5, 28.0], [76.5, 28.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"st_nm": "Karnataka"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[74.0, 12.0], [74.0, 18.0], [78.0, 18.0], [78.0, 12.0], [74.0, 12.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"st_nm": "Tamil Nadu"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[76.0, 8.0], [76.0, 13.0], [80.0, 13.0], [80.0, 8.0], [76.0, 8.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"st_nm": "Uttar Pradesh"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[77.0, 24.0], [77.0, 31.0], [84.0, 31.0], [84.0, 24.0], [77.0, 24.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"st_nm": "Gujarat"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[68.0, 20.0], [68.0, 25.0], [74.0, 25.0], [74.0, 20.0], [68.0, 20.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"st_nm": "Rajasthan"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[69.0, 23.0], [69.0, 30.0], [78.0, 30.0], [78.0, 23.0], [69.0, 23.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"st_nm": "West Bengal"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[85.0, 21.0], [85.0, 27.0], [89.0, 27.0], [89.0, 21.0], [85.0, 21.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"st_nm": "Madhya Pradesh"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[74.0, 21.0], [74.0, 26.0], [82.0, 26.0], [82.0, 21.0], [74.0, 21.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"st_nm": "Bihar"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[83.0, 24.0], [83.0, 27.0], [88.0, 27.0], [88.0, 24.0], [83.0, 24.0]]]
                }
            }
        ]
    }
    return india_geojson

def load_data_for_mapping():
    """Loads the cleaned data."""
    # Load cleaned crime data
    crime_df = pd.read_csv(DATA_PROCESSED_PATH / r"D:\crime-prediction-project\data\processed\cleaned_crime_data.csv")
    print("✅ Crime data loaded successfully!")
    
    # Create minimal GeoJSON
    india_geojson = create_minimal_geojson()
    print("✅ Minimal GeoJSON created for major Indian states!")
    
    return crime_df, india_geojson

def create_state_heatmap(crime_df, india_geojson):
    """Creates a heatmap of crimes by state."""
    print("Creating state-level heatmap...")
    
    # Count crimes by state
    state_crime_counts = crime_df['State'].value_counts().reset_index()
    state_crime_counts.columns = ['State', 'Crime_Count']
    
    # Create a base map centered on India
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='CartoDB positron')
    
    # Create a choropleth map to show crime density by state
    folium.Choropleth(
        geo_data=india_geojson,
        name='Choropleth',
        data=state_crime_counts,
        columns=['State', 'Crime_Count'],
        key_on='feature.properties.st_nm',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Number of Crime Incidents',
        highlight=True
    ).add_to(india_map)
    
    # Save the map
    state_map_path = APP_ASSETS_PATH / 'india_state_crime_heatmap.html'
    india_map.save(state_map_path)
    print(f"✅ State-level heatmap saved to: {state_map_path}")
    
    return india_map

def create_city_hotspot_map(crime_df):
    """Creates a detailed heatmap showing crime hotspots at the city level."""
    print("Creating city-level hotspot heatmap...")
    
    # Create a base map
    city_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    # Prepare data for heatmap (need latitude, longitude, and weight)
    heat_data = []
    for _, row in crime_df.iterrows():
        # Use severity score as weight, or just count as 1
        weight = row['Severity Score'] / 10  # Normalize severity for better visualization
        heat_data.append([row['Latitude'], row['Longitude'], weight])
    
    # Add heatmap to the map
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(city_map)
    
    # Save the map
    city_map_path = APP_ASSETS_PATH / 'india_city_crime_heatmap.html'
    city_map.save(city_map_path)
    print(f"✅ City-level hotspot map saved to: {city_map_path}")
    
    return city_map

def generate_hotspot_analysis(crime_df):
    """Generates a text report of top crime hotspots."""
    print("\n--- Detailed Hotspot Analysis Report ---")
    
    # Top 10 states by crime count
    top_states = crime_df['State'].value_counts().head(10)
    print("\nTop 10 States by Crime Count:")
    for state, count in top_states.items():
        print(f"  - {state}: {count} incidents")
    
    # Top 10 cities by crime count
    top_cities = crime_df['City'].value_counts().head(10)
    print("\nTop 10 Cities by Crime Count:")
    for city, count in top_cities.items():
        state = crime_df[crime_df['City'] == city]['State'].iloc[0]
        print(f"  - {city}, {state}: {count} incidents")
    
    # Most common crime types in top 3 cities
    print("\nMost Common Crime Types in Top 3 Cities:")
    for i, city in enumerate(top_cities.index[:3]):
        city_crimes = crime_df[crime_df['City'] == city]['Crime Type'].value_counts().head(3)
        print(f"\n{city}:")
        for crime_type, count in city_crimes.items():
            print(f"  - {crime_type}: {count} incidents")

if __name__ == '__main__':
    print("="*50)
    print("GEOSPATIAL ANALYSIS & HOTSPOT MAPPING")
    print("="*50)
    
    # Load data
    crime_df, india_geojson = load_data_for_mapping()
    
    if crime_df is not None and india_geojson is not None:
        # Create maps
        print("\n1. Creating state-level crime heatmap...")
        state_map = create_state_heatmap(crime_df, india_geojson)
        
        print("\n2. Creating city-level hotspot map...")
        city_map = create_city_hotspot_map(crime_df)
        
        print("\n3. Generating detailed hotspot analysis...")
        generate_hotspot_analysis(crime_df)
        
        print("\n✅ Geospatial analysis completed!")
        print("\n📊 Open the following files in your web browser to view the interactive maps:")
        print(f"   - State Heatmap: {APP_ASSETS_PATH / 'india_state_crime_heatmap.html'}")
        print(f"   - City Hotspots: {APP_ASSETS_PATH / 'india_city_crime_heatmap.html'}")