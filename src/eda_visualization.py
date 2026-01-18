import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set visual style for all plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Define file paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
DATA_PROCESSED_PATH = project_root / 'data' / 'processed'
REPORTS_FIGURES_PATH = project_root / 'reports' / 'figures'

def load_clean_data(filename='cleaned_crime_data.csv'):
    """Loads the cleaned dataset."""
    file_path = DATA_PROCESSED_PATH / filename
    print(f"Loading cleaned data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        # Fix the datetime conversion that failed in the previous step
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        print("✅ Cleaned data loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}. Please run data_preprocessing.py first.")
        return None

def plot_crime_type_distribution(df):
    """Objective 4: Plot distribution of crime types (Weapon Usage & Crime Outcome context)"""
    plt.figure(figsize=(14, 8))
    crime_counts = df['Crime Type'].value_counts()
    
    ax = sns.barplot(x=crime_counts.values, y=crime_counts.index, palette='viridis')
    plt.title('Distribution of Crime Types', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Incidents', fontsize=12)
    plt.ylabel('Crime Type', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(crime_counts.values):
        ax.text(v + 20, i, str(v), color='black', va='center')
    
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_PATH / 'crime_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_state_wise_crimes(df):
    """Objective 3: State-wise Crime Ranking"""
    plt.figure(figsize=(16, 10))
    state_crimes = df['State'].value_counts().head(15)  # Top 15 states
    
    ax = sns.barplot(x=state_crimes.values, y=state_crimes.index, palette='magma')
    plt.title('Top 15 States by Number of Crime Incidents', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Incidents', fontsize=12)
    plt.ylabel('State', fontsize=12)
    
    for i, v in enumerate(state_crimes.values):
        ax.text(v + 20, i, str(v), color='black', va='center')
    
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_PATH / 'state_wise_crimes.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_weapon_usage(df):
    """Objective 4: Weapon Usage Analysis"""
    plt.figure(figsize=(12, 6))
    weapon_counts = df['Weapon Type'].value_counts()
    
    plt.pie(weapon_counts.values, labels=weapon_counts.index, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Weapon Types Used in Crimes', fontsize=16, fontweight='bold')
    plt.savefig(REPORTS_FIGURES_PATH / 'weapon_usage_pie.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_crime_outcomes(df):
    """Objective 4: Crime Outcome Analysis"""
    plt.figure(figsize=(12, 6))
    outcome_counts = df['Outcome'].value_counts()
    
    ax = sns.barplot(x=outcome_counts.values, y=outcome_counts.index, palette='plasma')
    plt.title('Distribution of Crime Outcomes', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Incidents', fontsize=12)
    plt.ylabel('Outcome', fontsize=12)
    
    for i, v in enumerate(outcome_counts.values):
        ax.text(v + 20, i, str(v), color='black', va='center')
    
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_PATH / 'crime_outcomes.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_temporal_patterns(df):
    """Objective 2: Time-Based Crime Patterns"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Hourly distribution
    hourly_counts = df['Hour'].value_counts().sort_index()
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, ax=ax1, palette='coolwarm')
    ax1.set_title('Crime Distribution by Hour of Day', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Number of Incidents')
    
    # Daily distribution
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_counts = df['Day of Week'].value_counts().reindex(day_order)
    sns.barplot(x=daily_counts.index, y=daily_counts.values, ax=ax2, palette='viridis')
    ax2.set_title('Crime Distribution by Day of Week', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Number of Incidents')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_PATH / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_hotspot_reports(df):
    """Objective 9: Generate preliminary insights for hotspot reports"""
    print("\n--- Hotspot Analysis Preview ---")
    
    # Top 5 crime hotspots by city
    top_cities = df['City'].value_counts().head(5)
    print("\nTop 5 Cities by Crime Frequency:")
    for city, count in top_cities.items():
        print(f"  - {city}: {count} incidents")
    
    # Most common crime type in top city
    top_city = top_cities.index[0]
    top_crime = df[df['City'] == top_city]['Crime Type'].value_counts().index[0]
    print(f"\nMost common crime in {top_city}: {top_crime}")

if __name__ == '__main__':
    print("="*50)
    print("EXPLORATORY DATA ANALYSIS & VISUALIZATION")
    print("="*50)
    
    # Load the cleaned data
    crime_df = load_clean_data()
    
    if crime_df is not None:
        print(f"\nDataset Shape: {crime_df.shape}")
        
        # Generate all visualizations
        print("\n1. Analyzing crime type distribution...")
        plot_crime_type_distribution(crime_df)
        
        print("\n2. Analyzing state-wise crime rankings...")
        plot_state_wise_crimes(crime_df)
        
        print("\n3. Analyzing weapon usage patterns...")
        plot_weapon_usage(crime_df)
        
        print("\n4. Analyzing crime outcomes...")
        plot_crime_outcomes(crime_df)
        
        print("\n5. Analyzing temporal patterns...")
        plot_temporal_patterns(crime_df)
        
        print("\n6. Generating hotspot reports preview...")
        generate_hotspot_reports(crime_df)
        
        print("\n✅ All EDA visualizations completed and saved to 'reports/figures/'")