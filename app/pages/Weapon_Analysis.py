import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

# Add parent dir to path so we can import components
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from components.weapon_analytics import WeaponAnalyticsEngine

# Page configuration
st.set_page_config(
    page_title="Weapon-Based Crime Analysis",
    page_icon=None,
    layout="wide"
)

st.title("Weapon-Based Crime Analysis")
st.markdown("### Detailed analytics of weapon usage patterns using the actual crime dataset")


# ---------- DATA LOADER ----------

@st.cache_data(show_spinner=True)
def load_crime_data():
    # Primary dataset path (your absolute path)
    primary_path = r"D:\crime_pattern_prediction\data\processed\dataset.csv"

    # Keep some fallbacks in case you later move to relative paths / deployment
    possible_paths = [
        primary_path,
        "data/processed/cleaned_crime_data.csv",
        "data/processed_data.csv",
        "../data/processed_data.csv",
        "../../data/processed_data.csv",
        "processed_data.csv",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)

            # Ensure DateTime
            if "DateTime" in df.columns:
                df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
            else:
                # Try typical crime dataset columns
                if "DATE OCC" in df.columns and "TIME OCC" in df.columns:
                    df["DateTime"] = pd.to_datetime(
                        df["DATE OCC"].astype(str).str.strip() + " " +
                        df["TIME OCC"].astype(str).str.strip(),
                        errors="coerce"
                    )
                elif "Date" in df.columns and "Time" in df.columns:
                    df["DateTime"] = pd.to_datetime(
                        df["Date"].astype(str) + " " + df["Time"].astype(str),
                        errors="coerce"
                    )
                elif "DATE OCC" in df.columns:
                    df["DateTime"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
                else:
                    df["DateTime"] = pd.to_datetime("2000-01-01")

            # Minimal alignment with dataset (no randomness)

            # Crime Type
            if "Crime Type" not in df.columns:
                if "Crm Cd Desc" in df.columns:
                    df["Crime Type"] = df["Crm Cd Desc"].astype(str)
                else:
                    df["Crime Type"] = "Unknown"

            # State (use AREA NAME if present, else Unknown)
            if "State" not in df.columns:
                if "AREA NAME" in df.columns:
                    df["State"] = df["AREA NAME"].astype(str)
                else:
                    df["State"] = "Unknown"

            # Area Type (if missing, mark as Unknown)
            if "Area Type" not in df.columns:
                df["Area Type"] = "Unknown"

            # Weapon Type (use Weapon Desc if present)
            if "Weapon Type" not in df.columns:
                if "Weapon Desc" in df.columns:
                    df["Weapon Type"] = (
                        df["Weapon Desc"]
                        .fillna("Unknown")
                        .astype(str)
                        .str.strip()
                        .replace("", "Unknown")
                    )
                else:
                    df["Weapon Type"] = "Unknown"

            return df, path
    return None, None


def main():
    df, path = load_crime_data()

    if df is None or df.empty:
        st.error(
            "Could not find dataset. Please ensure "
            r"`D:\crime_pattern_prediction\data\processed\dataset.csv` exists."
        )
        return

    st.success(f"Loaded {len(df):,} records from `{path}`")

    # ---------- SIDEBAR FILTERS ----------

    st.sidebar.header("Weapon Analysis Filters")

    # Date range
    if "DateTime" in df.columns:
        min_date = df["DateTime"].min().date()
        max_date = df["DateTime"].max().date()
    else:
        min_date = datetime(2000, 1, 1).date()
        max_date = datetime(2000, 1, 1).date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Crime type filter
    if "Crime Type" in df.columns:
        crime_types = ["All"] + sorted(df["Crime Type"].dropna().unique().tolist())
    else:
        crime_types = ["All"]
    selected_crime_type = st.sidebar.selectbox("Crime Type", crime_types)

    # State filter
    if "State" in df.columns:
        states = ["All"] + sorted(df["State"].dropna().unique().tolist())
    else:
        states = ["All"]
    selected_state = st.sidebar.selectbox("State", states)

    # Area type filter
    if "Area Type" in df.columns:
        area_types = ["All"] + sorted(df["Area Type"].dropna().unique().tolist())
        selected_area = st.sidebar.selectbox("Area Type", area_types)
    else:
        selected_area = "All"

    # Weapon type filter
    if "Weapon Type" in df.columns:
        weapon_types = ["All"] + sorted(df["Weapon Type"].dropna().unique().tolist())
    else:
        weapon_types = ["All"]
    selected_weapon_type = st.sidebar.selectbox("Weapon Type", weapon_types)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced Options")
    min_weapon_count = st.sidebar.slider(
        "Minimum samples per weapon (for severity chart)",
        min_value=10,
        max_value=100,
        value=30,
        step=5
    )

    # ---------- APPLY FILTERS ----------

    df_filtered = df.copy()

    # Date filter
    if "DateTime" in df_filtered.columns:
        mask_date = (df_filtered["DateTime"].dt.date >= start_date) & (
            df_filtered["DateTime"].dt.date <= end_date
        )
        df_filtered = df_filtered[mask_date]

    if selected_crime_type != "All" and "Crime Type" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Crime Type"] == selected_crime_type]

    if selected_state != "All" and "State" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["State"] == selected_state]

    if selected_area != "All" and "Area Type" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Area Type"] == selected_area]

    if selected_weapon_type != "All" and "Weapon Type" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Weapon Type"] == selected_weapon_type]

    st.markdown(f"**Filtered Records:** {len(df_filtered):,}")

    if len(df_filtered) == 0:
        st.warning("No data for selected filters. Please adjust filters.")
        return

    # ---------- INIT ENGINE ----------
    engine = WeaponAnalyticsEngine(df_filtered)
    overall_stats = engine.get_overall_weapon_stats()

    # ---------- TOP METRICS ----------

    st.header("Weapon Usage Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Crimes (filtered)", f"{overall_stats['total_crimes']:,}")
    with col2:
        st.metric(
            "Crimes with Weapons",
            f"{overall_stats['with_weapon']:,}",
            delta=f"{overall_stats['weapon_usage_pct']}% of total"
        )
    with col3:
        if overall_stats["avg_severity_with_weapon"] is not None:
            st.metric(
                "Average Severity (Weapon)",
                f"{overall_stats['avg_severity_with_weapon']:.2f}"
            )
    with col4:
        if overall_stats["avg_severity_without_weapon"] is not None:
            st.metric(
                "Average Severity (No Weapon)",
                f"{overall_stats['avg_severity_without_weapon']:.2f}"
            )

    # ---------- MAIN VISUALIZATIONS ----------

    st.subheader("High-Level Weapon Analytics")

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = engine.fig_weapon_usage_pie()
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)

        fig_hour = engine.fig_hourly_pattern()
        if fig_hour:
            st.plotly_chart(fig_hour, use_container_width=True)

    with col2:
        fig_bar = engine.fig_weapon_type_bar()
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)

        fig_trend = engine.fig_weapon_usage_trend()
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True)

    # ---------- SEVERITY & RISK ----------

    st.subheader("Severity and Risk by Weapon Type")
    fig_sev = engine.fig_severity_by_weapon(min_count=min_weapon_count)
    if fig_sev:
        st.plotly_chart(fig_sev, use_container_width=True)
    else:
        st.info("Not enough data per weapon type for severity analysis with current filters.")

    # ---------- RELATIONSHIP ANALYSIS ----------

    st.subheader("Relationship Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Crime Type vs Weapon Type",
        "Area Type vs Weapon Type",
        "Top Weapons by State"
    ])

    with tab1:
        fig_heat_crime = engine.fig_crime_vs_weapon_heatmap()
        if fig_heat_crime:
            st.plotly_chart(fig_heat_crime, use_container_width=True)
        else:
            st.info("Not enough weapon data for Crime Type vs Weapon analysis.")

    with tab2:
        fig_heat_area = engine.fig_area_vs_weapon_heatmap()
        if fig_heat_area:
            st.plotly_chart(fig_heat_area, use_container_width=True)
        else:
            st.info("Not enough weapon data for Area Type vs Weapon analysis.")

    with tab3:
        fig_state = engine.fig_top_weapons_by_state()
        if fig_state:
            st.plotly_chart(fig_state, use_container_width=True)
        else:
            st.info("Not enough weapon data to compute state-wise weapon rankings.")

    # ---------- RAW DATA & EXPORT ----------

    with st.expander("View and Export Weapon-Filtered Data"):
        st.write("This table shows crimes where weapons were detected in the filtered subset.")

        # Use engine's processed frame and flag
        weapon_only = engine.df[engine.df["Weapon Used Flag"]].copy()

        # Choose safe columns based on what actually exists
        preferred_cols = [
            "DR_NO", "FIR Number", "Crime Type", "State", "City", "Area Type",
            "Weapon Used", "Weapon Used Cd", "Weapon Desc", "Weapon Type",
            "Severity Score", "DateTime"
        ]
        display_cols = [c for c in preferred_cols if c in weapon_only.columns]

        if not display_cols:
            st.info("No suitable columns available to display for weapon-based crimes.")
        else:
            st.dataframe(
                weapon_only[display_cols],
                use_container_width=True,
                height=350
            )

            # Download CSV
            csv_bytes = weapon_only[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Weapon-Based Crime Data (CSV)",
                data=csv_bytes,
                file_name="weapon_based_crimes_filtered.csv",
                mime="text/csv"
            )

    # ---------- EXPLANATION SECTION ----------

    with st.expander("How to Interpret These Weapon Analytics"):
        st.markdown("""
        This module uses only your dataset and provides:

        - Usage Overview: how often weapons are used and how that relates to overall crime volume.
        - Severity by Weapon Type: which weapons are associated with higher average severity (when a severity score exists).
        - Time Patterns: at what hours weapon-based crimes peak.
        - Geographical Patterns: which states or areas show higher concentration of weapon-related crimes.
        - Relationship Analysis:
            - Crime Type versus Weapon Type
            - Area Type versus Weapon Type
        - Temporal Trends: how weapon usage changes over time.

        All computations are done directly on your dataset; no simulated or random values are used.
        """)


if __name__ == "__main__":
    main()
