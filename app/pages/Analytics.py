# app/pages/analytics.py

import os
from typing import Dict, Any

import pandas as pd
import streamlit as st
import plotly.express as px

# Import the analytics engine from components
from components.analytics_engine import (
    ensure_canonical_columns,
    get_available_options,
    CrimeAnalyticsEngine,
    create_advanced_analytics_dashboard,
)

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Crime Pattern Analytics",
    page_icon=None,
    layout="wide",
)

st.title("Crime Pattern Analytics Dashboard")
st.markdown("This page provides advanced analytics directly from your crime dataset.")


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
DATA_PATH = r"D:\crime_pattern_prediction\data\processed\dataset.csv"


@st.cache_data(show_spinner=True)
def load_crime_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = ensure_canonical_columns(df)
    return df


try:
    df = load_crime_data(DATA_PATH)
    st.success(f"Loaded {len(df):,} records from: {DATA_PATH}")
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()


# ------------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------------
st.sidebar.header("Filters")

options: Dict[str, Any] = get_available_options(df)

# Year filter
year_options = ["All"] + [str(y) for y in options.get("years", [])]
selected_year = st.sidebar.selectbox("Year", year_options)

# Month filter
month_map = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
month_values = options.get("months", [])
month_labels = ["All"] + [f"{m:02d} - {month_map.get(m, str(m))}" for m in month_values]
selected_month_label = st.sidebar.selectbox("Month", month_labels)

# City filter
city_options = ["All"] + options.get("cities", [])
selected_city = st.sidebar.selectbox("City", city_options)

# Crime Type filter
crime_options = ["All"] + options.get("crime_types", [])
selected_crime = st.sidebar.selectbox("Crime Type", crime_options)

# Time category filter
tc_options = ["All"] + options.get("time_categories", [])
selected_time_cat = st.sidebar.selectbox("Time Category", tc_options)

# Part of day filter
pod_options = ["All"] + options.get("part_of_day", [])
selected_pod = st.sidebar.selectbox("Part of Day", pod_options)

# Weekend filter
weekend_filter = st.sidebar.selectbox(
    "Day Type",
    ["All", "Weekday only", "Weekend only"],
)


# ------------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------------
df_filtered = df.copy()

if selected_year != "All" and "Year" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Year"] == int(selected_year)]

if selected_month_label != "All" and "Month" in df_filtered.columns:
    sel_month_num = int(selected_month_label.split(" - ")[0])
    df_filtered = df_filtered[df_filtered["Month"] == sel_month_num]

if selected_city != "All" and "City" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["City"] == selected_city]

if selected_crime != "All" and "Crime Type" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Crime Type"] == selected_crime]

if selected_time_cat != "All" and "Time_Category" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Time_Category"] == selected_time_cat]

if selected_pod != "All" and "Part of Day" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Part of Day"] == selected_pod]

if weekend_filter != "All" and "Is_Weekend" in df_filtered.columns:
    if weekend_filter == "Weekday only":
        df_filtered = df_filtered[df_filtered["Is_Weekend"] == 0]
    elif weekend_filter == "Weekend only":
        df_filtered = df_filtered[df_filtered["Is_Weekend"] == 1]

st.markdown(f"**Filtered records:** {len(df_filtered):,}")

if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust filters.")
    st.stop()


# ------------------------------------------------------------------
# High-level metrics
# ------------------------------------------------------------------
st.header("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Crimes", f"{len(df_filtered):,}")

with col2:
    if "Severity Score" in df_filtered.columns:
        st.metric(
            "Average Severity",
            f"{df_filtered['Severity Score'].mean():.2f}",
        )

with col3:
    if "Crime Type" in df_filtered.columns:
        st.metric(
            "Unique Crime Types",
            f"{df_filtered['Crime Type'].nunique():,}",
        )

with col4:
    if "City" in df_filtered.columns:
        st.metric("Cities Covered", f"{df_filtered['City'].nunique():,}")


# ------------------------------------------------------------------
# Run analytics engine
# ------------------------------------------------------------------
engine = CrimeAnalyticsEngine(df_filtered)
dashboard = create_advanced_analytics_dashboard(engine)
insights = dashboard.get("raw_insights", {})


# ------------------------------------------------------------------
# Temporal analytics
# ------------------------------------------------------------------
st.header("Temporal Analytics")

temp_viz = dashboard.get("temporal_viz", {})

col_t1, col_t2 = st.columns(2)
with col_t1:
    fig_hourly = temp_viz.get("hourly_trend")
    if fig_hourly is not None:
        st.plotly_chart(fig_hourly, use_container_width=True)

with col_t2:
    fig_daily = temp_viz.get("daily_pattern")
    if fig_daily is not None:
        st.plotly_chart(fig_daily, use_container_width=True)

# Monthly trend (if available from seasonal viz)
seasonal_viz = dashboard.get("seasonal_viz", {})
fig_monthly = seasonal_viz.get("monthly_trends")
if fig_monthly is not None:
    st.plotly_chart(fig_monthly, use_container_width=True)


# ------------------------------------------------------------------
# Geospatial analytics
# ------------------------------------------------------------------
st.header("Geospatial Analytics")

geo_viz = dashboard.get("geospatial_viz", {})
col_g1, col_g2 = st.columns(2)

with col_g1:
    fig_states = geo_viz.get("state_chart")
    if fig_states is not None:
        st.plotly_chart(fig_states, use_container_width=True)

with col_g2:
    fig_cities = geo_viz.get("city_ranking")
    if fig_cities is not None:
        st.plotly_chart(fig_cities, use_container_width=True)

geo_insights = insights.get("geospatial_analysis", {})
clusters_df = geo_insights.get("clusters")
risk_density_df = geo_insights.get("risk_density")

with st.expander("Geographic clusters (top 10)"):
    if isinstance(clusters_df, pd.DataFrame) and not clusters_df.empty:
        st.dataframe(clusters_df)
    else:
        st.info("No geographic cluster information available for the current filters.")

with st.expander("High-risk grid cells (top 15 by risk score)"):
    if isinstance(risk_density_df, pd.DataFrame) and not risk_density_df.empty:
        st.dataframe(risk_density_df)
    else:
        st.info("No risk density information available for the current filters.")


# ------------------------------------------------------------------
# Crime type analytics
# ------------------------------------------------------------------
st.header("Crime Type Analytics")

crime_viz = dashboard.get("crime_type_viz", {})

col_c1, col_c2 = st.columns(2)
with col_c1:
    fig_crime_dist = crime_viz.get("crime_distribution")
    if fig_crime_dist is not None:
        st.plotly_chart(fig_crime_dist, use_container_width=True)

with col_c2:
    fig_severity = crime_viz.get("severity_analysis")
    if fig_severity is not None:
        st.plotly_chart(fig_severity, use_container_width=True)

crime_insights = insights.get("crime_type_analysis", {})
severity_by_crime = crime_insights.get("severity_analysis")

with st.expander("Crime types with severity statistics"):
    if isinstance(severity_by_crime, pd.DataFrame) and not severity_by_crime.empty:
        st.dataframe(severity_by_crime.head(20))
    else:
        st.info("No severity-by-crime statistics available for the current filters.")


# ------------------------------------------------------------------
# Seasonal / yearly analytics
# ------------------------------------------------------------------
st.header("Seasonal and Yearly Analytics")

fig_seasonal = seasonal_viz.get("seasonal_analysis")
if fig_seasonal is not None:
    st.plotly_chart(fig_seasonal, use_container_width=True)

seasonal_insights = insights.get("seasonal_analysis", {})
with st.expander("Raw seasonal statistics"):
    monthly_trends = seasonal_insights.get("monthly_trends")
    if isinstance(monthly_trends, pd.DataFrame) and not monthly_trends.empty:
        st.subheader("Monthly trends")
        st.dataframe(monthly_trends)
    yearly_trends = seasonal_insights.get("yearly_trends")
    if isinstance(yearly_trends, pd.DataFrame) and not yearly_trends.empty:
        st.subheader("Yearly trends")
        st.dataframe(yearly_trends)


# ------------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------------
st.header("Correlation Analysis")

corr_insights = insights.get("correlation_analysis", {})
numeric_corr = corr_insights.get("numeric_correlations")

if isinstance(numeric_corr, pd.DataFrame) and not numeric_corr.empty:
    st.subheader("Correlation between numeric variables")
    st.dataframe(numeric_corr.style.background_gradient(cmap="Reds"))
else:
    st.info("No numeric correlation matrix available for the current filters.")


# ------------------------------------------------------------------
# Raw data view
# ------------------------------------------------------------------
st.header("Filtered Data Preview")
st.dataframe(df_filtered.head(200))

csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered dataset as CSV",
    data=csv_bytes,
    file_name="crime_data_filtered.csv",
    mime="text/csv",
)
