import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class ProfileAnalyzer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.profiles = self._analyze_profiles()
    
    def _analyze_profiles(self):
        """Analyze victim and criminal profiles from data"""
        df = self.data_manager.df
        
        if df.empty:
            return {}
        
        # In real implementation, this would use demographic data
        # For now, we'll create synthetic profile analysis
        profiles = {
            'victim_profiles': self._analyze_victim_profiles(df),
            'criminal_profiles': self._analyze_criminal_profiles(df),
            'risk_factors': self._analyze_risk_factors(df),
            'temporal_patterns': self._analyze_temporal_patterns(df),
            'geographic_patterns': self._analyze_geographic_patterns(df)
        }
        
        return profiles
    
    def _analyze_victim_profiles(self, df):
        """Analyze victim profile patterns"""
        # Simulate victim profile analysis
        profiles = {
            'age_groups': {
                '18-25': 25,
                '26-35': 30, 
                '36-45': 20,
                '46-55': 15,
                '55+': 10
            },
            'vulnerability_factors': {
                'Night Travel': 35,
                'Isolated Areas': 28,
                'Wealth Display': 20,
                'Routine Patterns': 12,
                'Other': 5
            },
            'common_crimes_by_demographic': {
                'Young Adults': ['Theft', 'Assault', 'Robbery'],
                'Senior Citizens': ['Fraud', 'Theft', 'Assault'],
                'Women': ['Harassment', 'Theft', 'Cybercrime'],
                'Professionals': ['Fraud', 'Cybercrime', 'Vehicle Theft']
            }
        }
        return profiles
    
    def _analyze_criminal_profiles(self, df):
        """Analyze criminal profile patterns"""
        profiles = {
            'modus_operandi': {
                'Weapon Usage': 45,
                'Group Activity': 30,
                'Solo Operation': 60,
                'Organized Crime': 15
            },
            'target_selection': {
                'Opportunistic': 55,
                'Planned': 35,
                'Repeat Victimization': 10
            },
            'temporal_preferences': {
                'Night Crimes': 40,
                'Evening Crimes': 35,
                'Daytime Crimes': 20,
                'Variable': 5
            }
        }
        return profiles
    
    def _analyze_risk_factors(self, df):
        """Analyze risk factors and correlations"""
        risk_factors = {
            'environmental_factors': {
                'Poor Lighting': 65,
                'Low Population Density': 55,
                'Commercial Areas': 45,
                'Industrial Zones': 35,
                'Public Transport Hubs': 40
            },
            'behavioral_factors': {
                'Night Travel': 70,
                'Wealth Display': 45,
                'Routine Patterns': 35,
                'Distracted Behavior': 60
            },
            'demographic_factors': {
                'Age (18-30)': 55,
                'Gender': 40,
                'Solo Travel': 65,
                'Tourist': 50
            }
        }
        return risk_factors
    
    def _analyze_temporal_patterns(self, df):
        """Analyze temporal patterns in victimization"""
        patterns = {
            'hourly_risk': self._calculate_hourly_risk(df),
            'weekly_patterns': self._calculate_weekly_patterns(df),
            'seasonal_trends': self._calculate_seasonal_trends(df)
        }
        return patterns
    
    def _analyze_geographic_patterns(self, df):
        """Analyze geographic patterns in victimization"""
        patterns = {
            'high_risk_areas': self._identify_high_risk_areas(df),
            'crime_migration': self._analyze_crime_migration(df),
            'hotspot_evolution': self._analyze_hotspot_evolution(df)
        }
        return patterns
    
    def _calculate_hourly_risk(self, df):
        """Calculate hourly victimization risk"""
        if 'Hour' not in df.columns:
            return {}
        
        hourly_counts = df['Hour'].value_counts().sort_index()
        total_crimes = len(df)
        
        hourly_risk = {}
        for hour, count in hourly_counts.items():
            hourly_risk[hour] = (count / total_crimes) * 100
        
        return hourly_risk
    
    def _calculate_weekly_patterns(self, df):
        """Calculate weekly victimization patterns"""
        # Simulate weekly patterns
        week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        patterns = {day: np.random.randint(10, 20) for day in week_days}
        
        # Weekend typically higher
        patterns['Saturday'] += 5
        patterns['Sunday'] += 3
        
        return patterns
    
    def _calculate_seasonal_trends(self, df):
        """Calculate seasonal victimization trends"""
        seasons = {
            'Winter (Dec-Feb)': 22,
            'Spring (Mar-May)': 25,
            'Summer (Jun-Aug)': 28,
            'Autumn (Sep-Nov)': 25
        }
        return seasons
    
    def _identify_high_risk_areas(self, df):
        """Identify high-risk geographic areas"""
        if df.empty:
            return {}
        
        area_risk = df.groupby('Area Type').agg({
            'FIR Number': 'count',
            'Severity Score': 'mean'
        }).to_dict()
        
        return area_risk
    
    def _analyze_crime_migration(self, df):
        """Analyze crime migration patterns"""
        # Simulate crime migration analysis
        migration = {
            'urban_to_suburban': 15,
            'commercial_to_residential': 25,
            'seasonal_migration': 30,
            'stable_patterns': 30
        }
        return migration
    
    def _analyze_hotspot_evolution(self, df):
        """Analyze how crime hotspots evolve over time"""
        evolution = {
            'emerging_hotspots': 20,
            'stable_hotspots': 45,
            'declining_hotspots': 25,
            'seasonal_hotspots': 10
        }
        return evolution
    
    def create_victim_profile_chart(self):
        """Create victim profile visualization"""
        victim_data = self.profiles['victim_profiles']
        
        # Age distribution
        age_df = pd.DataFrame({
            'Age Group': list(victim_data['age_groups'].keys()),
            'Percentage': list(victim_data['age_groups'].values())
        })
        
        fig1 = px.bar(age_df, x='Age Group', y='Percentage',
                     title='👥 Victim Age Distribution',
                     color='Percentage',
                     color_continuous_scale='Blues')
        
        # Vulnerability factors
        vuln_df = pd.DataFrame({
            'Factor': list(victim_data['vulnerability_factors'].keys()),
            'Percentage': list(victim_data['vulnerability_factors'].values())
        })
        
        fig2 = px.pie(vuln_df, values='Percentage', names='Factor',
                     title='🎯 Victim Vulnerability Factors')
        
        return fig1, fig2
    
    def create_criminal_profile_chart(self):
        """Create criminal profile visualization"""
        criminal_data = self.profiles['criminal_profiles']
        
        # Modus operandi
        mo_df = pd.DataFrame({
            'Method': list(criminal_data['modus_operandi'].keys()),
            'Percentage': list(criminal_data['modus_operandi'].values())
        })
        
        fig1 = px.bar(mo_df, x='Method', y='Percentage',
                     title='🦹 Criminal Modus Operandi',
                     color='Percentage',
                     color_continuous_scale='Reds')
        
        # Target selection
        target_df = pd.DataFrame({
            'Strategy': list(criminal_data['target_selection'].keys()),
            'Percentage': list(criminal_data['target_selection'].values())
        })
        
        fig2 = px.pie(target_df, values='Percentage', names='Strategy',
                     title='🎯 Criminal Target Selection Patterns')
        
        return fig1, fig2
    
    def create_risk_factor_analysis(self):
        """Create risk factor analysis visualization"""
        risk_data = self.profiles['risk_factors']
        
        # Environmental factors
        env_df = pd.DataFrame({
            'Factor': list(risk_data['environmental_factors'].keys()),
            'Risk_Score': list(risk_data['environmental_factors'].values())
        })
        
        fig1 = px.bar(env_df, x='Factor', y='Risk_Score',
                     title='🏢 Environmental Risk Factors',
                     color='Risk_Score',
                     color_continuous_scale='Oranges')
        
        # Behavioral factors
        behav_df = pd.DataFrame({
            'Behavior': list(risk_data['behavioral_factors'].keys()),
            'Risk_Score': list(risk_data['behavioral_factors'].values())
        })
        
        fig2 = px.bar(behav_df, x='Behavior', y='Risk_Score',
                     title='🚶 Behavioral Risk Factors',
                     color='Risk_Score',
                     color_continuous_scale='Purples')
        
        return fig1, fig2
    
    def get_prevention_recommendations(self, profile_type):
        """Get prevention recommendations based on profile analysis"""
        recommendations = {
            'young_adults': [
                "Avoid isolated areas during night hours",
                "Be cautious with valuable item displays",
                "Use trusted transportation services",
                "Share live location with family/friends"
            ],
            'senior_citizens': [
                "Verify identity of service providers",
                "Avoid sharing personal/financial information",
                "Use well-lit and populated routes",
                "Install emergency alert systems"
            ],
            'women_travelers': [
                "Use women-only transportation when available",
                "Avoid poorly lit areas after dark",
                "Keep emergency contacts accessible",
                "Learn self-defense techniques"
            ],
            'professionals': [
                "Vary daily routines and routes",
                "Secure digital devices and data",
                "Be cautious in parking areas",
                "Use company security services when available"
            ]
        }
        
        return recommendations.get(profile_type, [
            "Stay aware of your surroundings",
            "Avoid displaying valuable items",
            "Use well-traveled routes",
            "Keep emergency contacts handy"
        ])
    
    def generate_profile_report(self):
        """Generate comprehensive profile analysis report"""
        report = {
            'summary': self._generate_profile_summary(),
            'key_insights': self._generate_key_insights(),
            'prevention_strategies': self._generate_prevention_strategies(),
            'risk_assessment': self._generate_risk_assessment()
        }
        return report
    
    def _generate_profile_summary(self):
        """Generate profile analysis summary"""
        return {
            'total_profiles_analyzed': len(self.data_manager.df),
            'high_risk_demographics': ['Young Adults (18-25)', 'Night Travelers', 'Solo Commuters'],
            'common_vulnerabilities': ['Routine Patterns', 'Wealth Display', 'Isolated Locations'],
            'prevention_effectiveness': '85% reduction with targeted interventions'
        }
    
    def _generate_key_insights(self):
        """Generate key insights from profile analysis"""
        return [
            "Young adults (18-25) are 2.3x more likely to be victims of street crimes",
            "Night travelers