import streamlit as st
import pandas as pd
from datetime import datetime

import os
import sys

# Add parent dir so we can import components
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from components.predictions import (
    DatabaseDrivenPredictionEngine,
    create_prediction_visualizations,
    generate_safety_recommendations,
)

# Page configuration
st.set_page_config(
    page_title="AI Crime Predictions - Real Data Analysis",
    page_icon=None,
    layout="wide"
)

st.title("AI Crime Prediction System")
st.markdown("### Real-time predictions based on actual crime records")

# -------------------------------------------------------------------
# MAIN PAGE LOGIC
# -------------------------------------------------------------------

def main():
    # Initialize engine with your dataset path
    engine = DatabaseDrivenPredictionEngine(
        csv_path=r"D:\crime_pattern_prediction\data\processed\dataset.csv"
    )

    if not engine.loaded:
        st.error(
            "Crime dataset not available.\n\n"
            r"Please ensure you have `D:\crime_pattern_prediction\data\processed\dataset.csv` "
            "or that your processor is correctly configured."
        )
        return

    # Get dataset insights from the engine
    insights = engine.get_dataset_insights()
    total_records = insights.get("total_records", 0)

    # Sidebar for input parameters
    st.sidebar.header("Location and Time Parameters")

    # Dataset overview in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Overview")
    if total_records:
        st.sidebar.write(f"Total Crimes: {total_records:,}")
    if insights.get("crime_types_count") is not None:
        st.sidebar.write(f"Crime Types: {insights['crime_types_count']}")
    if insights.get("states_count") is not None:
        st.sidebar.write(f"States Covered: {insights['states_count']}")
    if insights.get("severity_stats", {}).get("average") is not None:
        st.sidebar.write(
            f"Avg Severity: {insights['severity_stats']['average']:.1f}"
        )

    # Location inputs
    st.sidebar.subheader("Enter Coordinates")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        latitude = st.number_input(
            "Latitude",
            value=28.6139,
            format="%.6f",
            help="Enter latitude coordinate (e.g., 28.6139 for Delhi)",
        )
    with col2:
        longitude = st.number_input(
            "Longitude",
            value=77.2090,
            format="%.6f",
            help="Enter longitude coordinate (e.g., 77.2090 for Delhi)",
        )

    # Date and time inputs
    st.sidebar.subheader("Select Date and Time")
    selected_date = st.sidebar.date_input("Date", datetime.now())
    selected_time = st.sidebar.time_input("Time", datetime.now().time())

    # Derived temporal features
    selected_datetime = datetime.combine(selected_date, selected_time)
    hour = selected_datetime.hour
    month = selected_datetime.month
    day_of_week = selected_datetime.weekday()

    # Area type selection
    st.sidebar.subheader("Area Type")
    area_types = [
        "Residential Area",
        "Commercial Area",
        "Industrial Area",
        "Slum Area",
        "Park Area",
        "Transport Area",
    ]
    area_type = st.sidebar.selectbox("Select Area Type", area_types)

    # Display current selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Summary")
    st.sidebar.info(
        f"""
Parameters Set:
- Lat: {latitude}, Lon: {longitude}
- {selected_datetime.strftime('%Y-%m-%d %H:%M')}
- {area_type}
- {total_records:,} historical crimes
"""
    )

    # Prepare input data for engine
    input_data = {
        "latitude": latitude,
        "longitude": longitude,
        "hour": hour,
        "month": month,
        "day_of_week": day_of_week,
        "area_type": area_type,
    }

    # Generate predictions button
    if st.sidebar.button("Analyze with Real Data", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {total_records:,} crime records..."):

            # Use the unified engine
            prediction_results = engine.generate_comprehensive_prediction(input_data)

            risk_prediction = prediction_results.get("hotspot_risk", {})
            crime_predictions = prediction_results.get("crime_types", [])
            severity_prediction = prediction_results.get("severity", {})
            weapon_predictions = prediction_results.get("weapons", [])
            safety_score = prediction_results.get("safety_score", 0)

            # Display results
            st.header("Real Data Analysis Results")

            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Safety Score",
                    f"{safety_score:.0f}/100",
                    delta=f"Based on {total_records:,} crimes",
                )

            with col2:
                if "error" not in risk_prediction:
                    risk_level = risk_prediction.get("risk_level", "Unknown")
                    nearby_count = risk_prediction.get("nearby_crimes_count", 0)
                    st.metric(
                        "Risk Level",
                        risk_level,
                        delta=f"{nearby_count} nearby crimes",
                    )

            with col3:
                if "error" not in severity_prediction and severity_prediction:
                    severity_level = severity_prediction.get("severity_level", "Unknown")
                    severity_score = severity_prediction.get("severity_score", 0)
                    st.metric("Severity", f"{severity_level} ({severity_score})")

            with col4:
                if isinstance(crime_predictions, list) and crime_predictions:
                    top_crime = crime_predictions[0].get("crime_type", "Unknown")
                    top_prob = crime_predictions[0].get("probability", 0) * 100
                    st.metric("Top Crime", top_crime, delta=f"{top_prob:.1f}%")

            # Visualizations using shared helpers
            st.subheader("Data-Driven Analytics")

            figs = create_prediction_visualizations(prediction_results, insights)

            col1, col2 = st.columns(2)

            with col1:
                if "safety_gauge" in figs:
                    st.plotly_chart(figs["safety_gauge"], use_container_width=True)
                if "dataset_overview" in figs:
                    st.plotly_chart(figs["dataset_overview"], use_container_width=True)

            with col2:
                if "crime_probabilities" in figs:
                    st.plotly_chart(figs["crime_probabilities"], use_container_width=True)
                if "weapon_probabilities" in figs:
                    st.plotly_chart(figs["weapon_probabilities"], use_container_width=True)

            # Detailed Analysis
            st.subheader("Detailed Risk Analysis")

            if "error" not in risk_prediction:
                col1, col2 = st.columns(2)

                with col1:
                    st.info(
                        f"""
Risk Analysis:
- Overall Risk: {risk_prediction.get('risk_level', 'Unknown')}
- Risk Score: {risk_prediction.get('risk_score', 0):.1%}
- Nearby Crimes: {risk_prediction.get('nearby_crimes_count', 0)}
- Temporal Risk: {risk_prediction.get('temporal_risk', 0):.1%}
- Location Risk: {risk_prediction.get('location_risk', 0):.1%}
"""
                    )

                with col2:
                    risk_level = risk_prediction.get("risk_level", "Unknown")
                    if risk_level == "High":
                        st.error(
                            "High Risk Area - Historical data shows significant criminal activity"
                        )
                    elif risk_level == "Medium":
                        st.warning(
                            "Moderate Risk - Occasional criminal activity patterns detected"
                        )
                    else:
                        st.success(
                            "Low Risk - Minimal historical criminal activity"
                        )

            # Crime Type Details
            if isinstance(crime_predictions, list) and crime_predictions:
                st.write("Crime Type Predictions:")
                for i, crime in enumerate(crime_predictions, 1):
                    prob_percent = crime.get("probability", 0) * 100
                    st.progress(
                        int(prob_percent),
                        text=f"{crime.get('crime_type', 'Unknown')} - {prob_percent:.1f}% probability",
                    )

            # Safety Recommendations
            st.subheader("Data-Driven Safety Recommendations")
            recommendations = generate_safety_recommendations(prediction_results, input_data)
            for i, tip in enumerate(recommendations, 1):
                st.write(f"{i}. {tip}")

    # Dataset Information
    with st.expander("Dataset Information and Methodology"):
        st.markdown(
            f"""
Real Data Analysis Methodology:
        
This system analyzes {total_records:,} actual crime records using:
        
- Geospatial Analysis: crimes within a defined radius
- Temporal Patterns: hour, day, and month-based analysis
- Statistical Modeling: probability calculations from historical data
- Pattern Recognition: area-specific crime distributions
        
Top 5 Crime Types:
"""
        )

        top_crimes = insights.get("top_crimes", {})
        for crime, count in top_crimes.items():
            st.write(
                f"- {crime}: {count:,} incidents ({count / total_records * 100:.1f}%)"
            )

        top_states = insights.get("top_states", {})
        if top_states:
            st.markdown("Top 5 States by Crime Count:")
            for state, count in top_states.items():
                st.write(f"- {state}: {count:,} crimes")


if __name__ == "__main__":
    main()
