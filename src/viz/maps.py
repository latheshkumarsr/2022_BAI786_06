
import folium
def create_map(center=(34.05,-118.25), zoom_start=12):
    m = folium.Map(location=center, zoom_start=zoom_start)
    return m
