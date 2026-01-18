import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def generate_detailed_hotspot_reports():
    """Generate detailed PDF-style reports for each major crime hotspot."""
    
    # Load data
    df = pd.read_csv(Path(__file__).parent.parent / 'data' / 'processed' / 'cleaned_crime_data.csv')
    
    # Get top 10 cities by crime count
    top_cities = df['City'].value_counts().head(10).index.tolist()
    
    reports = []
    
    for city in top_cities:
        city_data = df[df['City'] == city]
        state = city_data['State'].iloc[0]
        
        # Generate report for this city
        report = {
            'City': city,
            'State': state,
            'Total_Incidents': len(city_data),
            'Most_Common_Crime': city_data['Crime Type'].value_counts().index[0],
            'Most_Common_Weapon': city_data['Weapon Type'].value_counts().index[0],
            'Most_Common_Outcome': city_data['Outcome'].value_counts().index[0],
            'Avg_Severity_Score': city_data['Severity Score'].mean(),
            'Peak_Hour': city_data['Hour'].value_counts().index[0],
            'Peak_Day': city_data['Day of Week'].value_counts().index[0],
            'Top_Area_Type': city_data['Area Type'].value_counts().index[0],
            'Report_Date': datetime.now().strftime("%Y-%m-%d")
        }
        
        reports.append(report)
    
    # Create DataFrame and save as CSV (can be extended to PDF later)
    report_df = pd.DataFrame(reports)
    output_path = Path(__file__).parent.parent / 'reports' / 'hotspot_reports' / 'detailed_hotspot_analysis.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_df.to_csv(output_path, index=False)
    print(f"✅ Detailed hotspot reports saved to: {output_path}")
    
    return report_df

if __name__ == "__main__":
    print("Generating detailed hotspot reports...")
    reports = generate_detailed_hotspot_reports()
    print("\n=== HOTSPOT REPORTS SUMMARY ===")
    print(reports[['City', 'State', 'Total_Incidents', 'Most_Common_Crime', 'Avg_Severity_Score']])