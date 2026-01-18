# src/route_visualization.py
import folium
from folium.plugins import MarkerCluster, HeatMap

class RouteVisualizer:
    def __init__(self, route_planner):
        self.planner = route_planner
    
    def create_interactive_route_map(self, route_analysis, map_center=None):
        """Create interactive Folium map with route safety visualization"""
        if not map_center:
            map_center = route_analysis['origin']
        
        m = folium.Map(location=map_center, zoom_start=13)
        
        # Add crime heatmap layer
        crime_heatmap = self._create_crime_heatmap_layer()
        crime_heatmap.add_to(m)
        
        # Plot recommended route
        self._plot_route_with_safety(m, route_analysis['recommended_route'], 'green', 'Safest Route')
        
        # Plot alternative routes
        colors = ['blue', 'orange', 'purple']
        for i, alt_route in enumerate(route_analysis['alternative_routes']):
            self._plot_route_with_safety(m, alt_route, colors[i], f'Alternative {i+1}')
        
        # Add risk markers
        self._add_risk_markers(m, route_analysis['recommended_route'])
        
        # Add safety legend
        self._add_safety_legend(m)
        
        return m
    
    def _plot_route_with_safety(self, map_obj, route, color, route_name):
        """Plot route with color-coded safety segments"""
        coordinates = route['coordinates']
        
        # Create color gradient based on safety
        for i in range(len(coordinates)-1):
            segment_start = coordinates[i]
            segment_end = coordinates[i+1]
            
            # Get safety score for this segment
            segment_safety = self._get_segment_safety(segment_start, segment_end, route)
            segment_color = self._safety_score_to_color(segment_safety)
            
            folium.PolyLine(
                [segment_start, segment_end],
                color=segment_color,
                weight=6,
                opacity=0.8,
                popup=f"Safety Score: {segment_safety}/100"
            ).add_to(map_obj)
        
        # Add route start/end markers
        folium.Marker(
            coordinates[0],
            popup=f"Start: {route_name}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(map_obj)
        
        folium.Marker(
            coordinates[-1],
            popup=f"End: {route_name}",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(map_obj)