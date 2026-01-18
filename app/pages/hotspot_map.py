import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
from datetime import datetime
import os

st.set_page_config(page_title="Live Crime Map", layout="wide")

st.title("Live Crime Map and Hotspots")


DATA_PATH = r"D:\crime_pattern_prediction\data\processed\dataset.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load CSV with some robustness to column name variations."""
    possible_paths = [
        DATA_PATH,
        "data/processed/dataset.csv",
        "data/processed/cleaned_crime_data.csv",
        "data/processed_data.csv",
        "../data/processed_data.csv",
        "../../data/processed_data.csv",
        "processed_data.csv",
    ]

    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            break

    if df is None:
        raise FileNotFoundError(
            f"Could not find dataset. Checked paths: {possible_paths}"
        )

    # Standardize column names to strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # Common name mappings (adjust as necessary)
    rename_map = {
        "LAT": "Latitude",
        "LON": "Longitude",
        "Lon": "Longitude",
        "lat": "Latitude",
        "lon": "Longitude",
        "Crm Cd Desc": "Crime Type",
        "Crm Cd": "Crm Code",
        "DATE OCC": "Date Occurred",
        "Date Rptd": "Date Reported",
        "TIME OCC": "Time Occurred",
        "occ_year": "occ_year",
        "occ_month": "occ_month",
        "occ_date": "occ_date",
        "occ_day": "occ_day",
    }

    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Ensure numeric coordinates
    for coord in ["Latitude", "Longitude"]:
        if coord in df.columns:
            df[coord] = pd.to_numeric(df[coord], errors="coerce")

    # Normalize Crime Type
    if "Crime Type" in df.columns:
        df["Crime Type"] = df["Crime Type"].astype(str).str.strip()
    elif "Crm Code" in df.columns:
        df["Crime Type"] = df["Crm Code"].astype(str)

    # Time processing
    time_candidates = [c for c in df.columns if "time" in c.lower()]
    date_candidates = [
        c for c in df.columns if "date" in c.lower() or "occurred" in c.lower()
    ]

    # Prefer 'Date Occurred' then others
    date_col = None
    for name in ("Date Occurred", "Date Reported", *date_candidates):
        if name in df.columns:
            date_col = name
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["_date_col_used"] = date_col
    else:
        df["_date_col_used"] = None

    # Extract hour if possible
    for t in ("Time Occurred", *time_candidates):
        if t in df.columns:
            def parse_hour(val):
                try:
                    if pd.isna(val):
                        return np.nan
                    s = str(val).strip()
                    if ":" in s:
                        return int(pd.to_datetime(s).hour)
                    if s.isdigit():
                        s = s.zfill(4)
                        return int(s[:2])
                    return np.nan
                except Exception:
                    return np.nan

            df["Hour"] = df[t].apply(parse_hour)
            break

    # If Hour not extracted but datetime exists, use its hour
    if (
        "Hour" not in df.columns
        and date_col
        and pd.api.types.is_datetime64_any_dtype(df[date_col])
    ):
        df["Hour"] = df[date_col].dt.hour

    # Fill a simple Severity Score if not present
    if "Severity Score" not in df.columns:
        sev_map_keywords = {
            "murder": 10,
            "homicide": 10,
            "assault": 7,
            "robbery": 8,
            "burglary": 6,
            "theft": 3,
            "vehicle theft": 5,
            "vehicle": 4,
            "vandalism": 2,
            "drug": 5,
            "fraud": 4,
            "cyber": 2,
        }

        def infer_severity(crime):
            if pd.isna(crime):
                return 1
            s = str(crime).lower()
            for k, v in sev_map_keywords.items():
                if k in s:
                    return v
            return 3

        if "Crime Type" in df.columns:
            df["Severity Score"] = df["Crime Type"].apply(infer_severity)
        else:
            df["Severity Score"] = 3

    # Weapon Type: derive from dataset only (no randomness)
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
    else:
        df["Weapon Type"] = (
            df["Weapon Type"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    # State column fallback
    if "State" not in df.columns:
        if "AREA NAME" in df.columns:
            df["State"] = df["AREA NAME"].astype(str)
        else:
            df["State"] = "Unknown"

    return df


def create_interactive_map(df: pd.DataFrame, map_type: str = "heatmap") -> folium.Map:
    """Create interactive Folium map centered on the data and add heat or cluster markers."""
    if df["Latitude"].notna().any() and df["Longitude"].notna().any():
        center = [df["Latitude"].median(), df["Longitude"].median()]
        zoom = 10 if len(df) < 1000 else 6
    else:
        center = [20.5937, 78.9629]
        zoom = 5

    m = folium.Map(location=center, zoom_start=zoom)

    if map_type == "heatmap":
        heat_data = [
            [
                row["Latitude"],
                row["Longitude"],
                float(row["Severity Score"]) if pd.notna(row["Severity Score"]) else 1,
            ]
            for _, row in df.iterrows()
            if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"])
        ]

        if heat_data:
            HeatMap(heat_data, min_opacity=0.2, radius=12, blur=8).add_to(m)

    elif map_type == "cluster":
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in df.iterrows():
            if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
                sev = row.get("Severity Score", 1)
                color = "red" if sev >= 8 else ("orange" if sev >= 5 else "green")

                date_col_name = row.get("_date_col_used")
                date_value = (
                    row.get(date_col_name)
                    if isinstance(date_col_name, str) and date_col_name in df.columns
                    else ""
                )

                hour_val = row.get("Hour")
                hour_str = int(hour_val) if pd.notna(hour_val) else "N/A"

                popup_text = (
                    f"<b>Crime Type:</b> {row.get('Crime Type', 'N/A')}<br>"
                    f"<b>Severity:</b> {sev}<br>"
                    f"<b>Weapon:</b> {row.get('Weapon Type', 'N/A')}<br>"
                    f"<b>Date:</b> {date_value}<br>"
                    f"<b>Hour:</b> {hour_str}"
                )
                folium.Marker(
                    [row["Latitude"], row["Longitude"]],
                    popup=folium.Popup(popup_text, max_width=350),
                    tooltip=str(row.get("Crime Type", "Crime")),
                    icon=folium.Icon(color=color),
                ).add_to(marker_cluster)

    return m


def safe_date_filter(df: pd.DataFrame, date_column: str, date_range) -> pd.DataFrame:
    """Safely filter dataframe by date range (expects date_range as [start, end] date objects)."""
    if not date_column or date_range is None or len(date_range) != 2:
        return df

    try:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
        return df.loc[mask]

    except Exception as e:
        st.error(f"Error in date filtering: {e}")
        return df


def main():
    st.sidebar.header("Map Filters")

    try:
        df = load_data()
    except Exception as e:
        st.sidebar.error(f"Failed to load data: {e}")
        return

    required_columns = [
        "Crime Type",
        "Severity Score",
        "Latitude",
        "Longitude",
        "Weapon Type",
    ]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.write(
            "Provide a CSV with at least these columns or allow the loader to infer them from similar names."
        )
        st.stop()

    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    if df.empty:
        st.warning("No records with valid coordinates found in the dataset.")
        st.stop()

    # Date column detection
    date_col = df.get("_date_col_used") if "_date_col_used" in df.columns else None
    if isinstance(date_col, pd.Series):
        date_col = date_col.iloc[0]

    if not isinstance(date_col, str) or date_col not in df.columns:
        date_col = None
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break

    crime_types = ["All"] + sorted(df["Crime Type"].dropna().unique().tolist())
    selected_crime = st.sidebar.selectbox("Crime Type", crime_types)

    # Date range
    try:
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            min_date = df[date_col].min().date()
            max_date = df[date_col].max().date()
            date_range = st.sidebar.date_input(
                "Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
            )
        else:
            date_range = None
    except Exception as e:
        st.sidebar.error(f"Date processing error: {e}")
        date_range = None

    # Severity slider
    min_sev = int(df["Severity Score"].min())
    max_sev = int(df["Severity Score"].max())
    min_severity, max_severity = st.sidebar.slider(
        "Severity Range", min_value=min_sev, max_value=max_sev, value=(min_sev, max_sev)
    )

    # State filter
    states = ["All"] + sorted(df["State"].dropna().unique().tolist()) if "State" in df.columns else ["All"]
    selected_state = st.sidebar.selectbox("State", states)

    # Map type
    map_type_choice = st.sidebar.radio("Map Type", ["Heatmap", "Cluster Markers"])

    # Apply filters
    filtered_df = df.copy()
    if selected_crime != "All":
        filtered_df = filtered_df[filtered_df["Crime Type"] == selected_crime]

    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["State"] == selected_state]

    filtered_df = filtered_df[
        (filtered_df["Severity Score"] >= min_severity)
        & (filtered_df["Severity Score"] <= max_severity)
    ]

    if date_range and date_col:
        filtered_df = safe_date_filter(filtered_df, date_col, date_range)

    st.subheader(f"Showing {len(filtered_df)} crimes")

    if filtered_df.empty:
        st.warning(
            "No data available for the selected filters. Please adjust your filters."
        )
        st.stop()

    # Map
    map_obj = create_interactive_map(
        filtered_df, "heatmap" if map_type_choice == "Heatmap" else "cluster"
    )
    st_folium(map_obj, width=1200, height=600)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Crimes", len(filtered_df))

    with col2:
        avg_severity = filtered_df["Severity Score"].mean()
        st.metric("Average Severity", f"{avg_severity:.1f}")

    with col3:
        top_crime = filtered_df["Crime Type"].mode()
        top_crime_value = top_crime.iloc[0] if not top_crime.empty else "N/A"
        st.metric("Most Common Crime", top_crime_value)

    with col4:
        weapons_count = filtered_df["Weapon Type"].nunique()
        st.metric("Weapon Types Used", weapons_count)

    # Quick insights
    st.subheader("Quick Insights")
    col1, col2 = st.columns(2)

    with col1:
        crime_dist = filtered_df["Crime Type"].value_counts().head(10)
        fig1 = px.bar(
            x=crime_dist.values,
            y=crime_dist.index,
            orientation="h",
            title="Top Crime Types",
            labels={"x": "Count", "y": "Crime Type"},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        if "Hour" in filtered_df.columns:
            hourly_pattern = (
                filtered_df["Hour"].dropna().astype(int).value_counts().sort_index()
            )
            fig2 = px.line(
                x=hourly_pattern.index,
                y=hourly_pattern.values,
                title="Crimes by Hour of Day",
                labels={"x": "Hour", "y": "Number of Crimes"},
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Show data sample and allow download
    st.subheader("Sample of Filtered Data")
    st.dataframe(filtered_df.head(200))

    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_crime_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
