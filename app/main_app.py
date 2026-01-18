# main_app.py - Crime Pattern Analysis System (root dashboard)

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.analytics_engine import ensure_canonical_columns

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Crime Pattern Analysis System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------------
# Minimal CSS (no top navigation bar)
# -------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metric cards */
    .metric-card {
        background: #1a1d29;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }

    /* Alert styles */
    .alert-danger {
        background: linear-gradient(90deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 107, 107, 0.05) 100%);
        border: 1px solid rgba(255, 107, 107, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .alert-warning {
        background: linear-gradient(90deg, rgba(255, 165, 0, 0.1) 0%, rgba(255, 165, 0, 0.05) 100%);
        border: 1px solid rgba(255, 165, 0, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Slight top padding so content is not glued to top */
    .main .block-container {
        padding-top: 40px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Data loading helpers (real dataset only)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_crime_data() -> pd.DataFrame:
    """
    Load the real processed dataset and canonicalize columns.
    No synthetic/sample data is generated here: everything comes
    from the actual CSV.
    """
    possible_paths = [
        r"D:\crime_pattern_prediction\data\processed\dataset.csv",
        "data/processed/dataset.csv",
        "../data/processed/dataset.csv",
        "dataset.csv",
    ]

    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break

    if csv_path is None:
        raise FileNotFoundError(
            "Could not find dataset. Expected at one of:\n"
            + "\n".join(possible_paths)
        )

    df = pd.read_csv(csv_path, low_memory=False)
    df = ensure_canonical_columns(df)

    # Ensure a pure date column exists for sidebar info and trends
    if "DateTime" in df.columns:
        dt = pd.to_datetime(df["DateTime"], errors="coerce")
        df["Date"] = dt.dt.date
    elif "Date_parsed" in df.columns:
        dp = pd.to_datetime(df["Date_parsed"], errors="coerce")
        df["Date"] = dp.dt.date
    else:
        df["Date"] = pd.NaT

    # Weapon usage flag (for internal analytics, not the removed metric card)
    if "Weapon Used" in df.columns:
        df["Weapon Used Flag"] = (
            df["Weapon Used"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["yes", "y", "true", "1"])
        )
    elif "Weapon Type" in df.columns:
        df["Weapon Used Flag"] = ~df["Weapon Type"].astype(str).str.lower().isin(
            ["none", "unknown", ""]
        )
    else:
        df["Weapon Used Flag"] = False

    df["Is_Weapon"] = df["Weapon Used Flag"].astype(int)

    # Violent crime flag based on Crime Type text (deterministic rules)
    def is_violent(crime):
        if pd.isna(crime):
            return False
        s = str(crime).lower()
        violent_keywords = [
            "murder",
            "homicide",
            "assault",
            "robbery",
            "rape",
            "sexual",
            "kidnap",
            "kidnapping",
            "rioting",
            "attempt to murder",
        ]
        return any(k in s for k in violent_keywords)

    if "Is_Violent" not in df.columns:
        df["Is_Violent"] = df["Crime Type"].apply(is_violent).astype(int)

    return df


# -------------------------------------------------------------------
# Dashboard class
# -------------------------------------------------------------------
class CrimeDashboard:
    def __init__(self):
        try:
            self.df = load_crime_data()
            st.success(
                f"Loaded {len(self.df):,} crime records from your processed dataset"
            )
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            self.df = pd.DataFrame()

    # ---------------- Header ----------------
    def render_header(self):
        st.markdown(
            """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #635bff; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                Crime Pattern Analysis Dashboard
            </h1>
            <p style="color: #b0b0b0; font-size: 1rem;">
                Advanced analysis of crime patterns using your processed crime dataset
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ---------------- Metrics ----------------
    def render_metrics(self):
        if self.df.empty:
            st.warning("No data available to compute metrics.")
            return

        total_crimes = len(self.df)

        # Average severity
        if "Severity Score" in self.df.columns:
            avg_severity = float(self.df["Severity Score"].mean())
        else:
            avg_severity = None

        # Violent crimes (based on text rules)
        violent_rate = float(self.df["Is_Violent"].mean() * 100) if "Is_Violent" in self.df.columns else None

        # Coverage (states or cities)
        if "State" in self.df.columns:
            coverage_label = "States Covered"
            coverage_value = self.df["State"].nunique()
        elif "City" in self.df.columns:
            coverage_label = "Cities Covered"
            coverage_value = self.df["City"].nunique()
        else:
            coverage_label = "Unique Locations"
            coverage_value = self.df["Location"].nunique() if "Location" in self.df.columns else 0

        col1, col2, col3, col4 = st.columns(4)

        # 1. Total crimes
        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div style="color: #b0b0b0; font-size: 0.9rem;">Total Crimes</div>
                <div style="font-size: 2rem; font-weight: 700; color: #635bff;">{total_crimes:,}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # 2. Average severity
        with col2:
            if avg_severity is not None:
                sev_text = f"{avg_severity:.1f}"
            else:
                sev_text = "N/A"
            st.markdown(
                f"""
            <div class="metric-card">
                <div style="color: #b0b0b0; font-size: 0.9rem;">Average Severity</div>
                <div style="font-size: 2rem; font-weight: 700; color: #00cc96;">{sev_text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # 3. Violent crime share (replaces weapon involvement card)
        with col3:
            if violent_rate is not None:
                violent_text = f"{violent_rate:.1f}%"
            else:
                violent_text = "N/A"
            st.markdown(
                f"""
            <div class="metric-card">
                <div style="color: #b0b0b0; font-size: 0.9rem;">Violent Crime Share</div>
                <div style="font-size: 2rem; font-weight: 700; color: #ffa500;">{violent_text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # 4. Coverage
        with col4:
            st.markdown(
                f"""
            <div class="metric-card">
                <div style="color: #b0b0b0; font-size: 0.9rem;">{coverage_label}</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #ffffff;">{coverage_value}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ---------------- Crime type chart ----------------
    def render_crime_type_chart(self):
        if "Crime Type" not in self.df.columns:
            return

        crime_counts = self.df["Crime Type"].value_counts().head(10)
        if crime_counts.empty:
            return

        fig = px.bar(
            x=crime_counts.values,
            y=crime_counts.index,
            orientation="h",
            title="Top 10 Crime Types",
            color=crime_counts.values,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            xaxis_title="Number of Crimes",
            yaxis_title="Crime Type",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Temporal / area analysis ----------------
    def render_time_analysis(self):
        col1, col2 = st.columns(2)

        # Crimes by hour
        with col1:
            if "Hour" in self.df.columns:
                hour_series = self.df["Hour"].dropna().astype(int)
                if not hour_series.empty:
                    hourly_counts = hour_series.value_counts().sort_index()
                    fig = px.line(
                        x=hourly_counts.index,
                        y=hourly_counts.values,
                        title="Crimes by Hour of Day",
                        markers=True,
                    )
                    fig.update_layout(
                        xaxis_title="Hour",
                        yaxis_title="Number of Crimes",
                        height=300,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Crimes by top location (State or City)
        with col2:
            if "State" in self.df.columns:
                grp_col = "State"
            elif "City" in self.df.columns:
                grp_col = "City"
            else:
                grp_col = None

            if grp_col is not None:
                counts = self.df[grp_col].value_counts().head(10)
                if not counts.empty:
                    fig = px.bar(
                        x=counts.index,
                        y=counts.values,
                        title=f"Top 10 {grp_col}s by Crime Count",
                        color=counts.values,
                        color_continuous_scale="Plasma",
                    )
                    fig.update_layout(
                        xaxis_title=grp_col,
                        yaxis_title="Number of Crimes",
                        height=300,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Data preview ----------------
    def render_data_preview(self):
        st.subheader("Data Preview")

        st.info(
            f"Dataset shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns"
        )

        col1, col2, col3 = st.columns(3)

        # State / City filter
        with col1:
            loc_label = None
            loc_choices = ["All"]
            if "State" in self.df.columns:
                loc_label = "State"
            elif "City" in self.df.columns:
                loc_label = "City"

            if loc_label is not None:
                loc_values = (
                    self.df[loc_label].dropna().astype(str).sort_values().unique().tolist()
                )
                loc_choices += loc_values
                selected_loc = st.selectbox(
                    f"Filter by {loc_label}", options=loc_choices
                )
            else:
                selected_loc = "All"

        # Crime type filter
        with col2:
            if "Crime Type" in self.df.columns:
                crime_values = (
                    self.df["Crime Type"]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .index[:20]
                    .tolist()
                )
                crime_choices = ["All Crimes"] + crime_values
                selected_crime = st.selectbox(
                    "Filter by Crime Type", options=crime_choices
                )
            else:
                selected_crime = "All Crimes"

        with col3:
            rows_to_show = st.slider("Rows to display", 10, 200, 50)

        filtered_df = self.df.copy()

        if (
            ("State" in self.df.columns or "City" in self.df.columns)
            and selected_loc != "All"
        ):
            if "State" in self.df.columns and selected_loc in self.df["State"].values:
                filtered_df = filtered_df[filtered_df["State"] == selected_loc]
            elif "City" in self.df.columns and selected_loc in self.df["City"].values:
                filtered_df = filtered_df[filtered_df["City"] == selected_loc]

        if "Crime Type" in self.df.columns and selected_crime != "All Crimes":
            filtered_df = filtered_df[filtered_df["Crime Type"] == selected_crime]

        st.dataframe(
            filtered_df.head(rows_to_show),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered data as CSV",
            data=csv_bytes,
            file_name="crime_data_filtered_root_dashboard.csv",
            mime="text/csv",
        )

    # ---------------- Map ----------------
    def render_map(self):
        if "Latitude" not in self.df.columns or "Longitude" not in self.df.columns:
            st.info("No latitude/longitude columns available to render map.")
            return

        st.subheader("Crime Location Map")

        df_map = self.df.dropna(subset=["Latitude", "Longitude"]).copy()
        if df_map.empty:
            st.info("No valid coordinates in dataset.")
            return

        # Downsample for performance (deterministic)
        max_points = 2000
        if len(df_map) > max_points:
            df_map = df_map.sample(n=max_points, random_state=42)

        hover_cols = []
        if "Crime Type" in df_map.columns:
            hover_cols.append("Crime Type")
        if "State" in df_map.columns:
            hover_cols.append("State")
        if "City" in df_map.columns:
            hover_cols.append("City")
        if "DateTime" in df_map.columns:
            hover_cols.append("DateTime")

        fig = px.scatter_mapbox(
            df_map,
            lat="Latitude",
            lon="Longitude",
            hover_name="Crime Type" if "Crime Type" in df_map.columns else None,
            hover_data=hover_cols if hover_cols else None,
            color_discrete_sequence=["#ff6b6b"],
            zoom=4,
            height=500,
        )

        fig.update_layout(
            mapbox_style="carto-darkmatter",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Upload section ----------------
    def render_upload_section(self):
        st.subheader("Upload Another Crime Dataset")

        uploaded_file = st.file_uploader(
            "Upload processed crime data CSV",
            type=["csv"],
            help="Columns will be normalized automatically using the analytics engine.",
        )

        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file, low_memory=False)
                df_uploaded = ensure_canonical_columns(df_uploaded)
                if "DateTime" in df_uploaded.columns:
                    dt = pd.to_datetime(df_uploaded["DateTime"], errors="coerce")
                    df_uploaded["Date"] = dt.dt.date
                st.success(f"File uploaded: {len(df_uploaded):,} rows")

                with st.expander("Preview uploaded data"):
                    st.dataframe(df_uploaded.head(), use_container_width=True)

                if st.button("Use this uploaded dataset in this session"):
                    self.df = df_uploaded
                    st.success("Switched to uploaded dataset for this session.")
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # ---------------- Run dashboard ----------------
    def run(self):
        if self.df is None or self.df.empty:
            st.error(
                "Dataset is empty or not loaded. Please ensure the processed dataset exists."
            )
            return

        # Sidebar dataset info
        st.sidebar.subheader("Dataset Info")
        st.sidebar.write(f"Rows: {len(self.df):,}")
        st.sidebar.write(f"Columns: {len(self.df.columns)}")

        if "Date" in self.df.columns:
            try:
                date_series = pd.to_datetime(self.df["Date"], errors="coerce")
                if not date_series.isna().all():
                    min_d = date_series.min().date()
                    max_d = date_series.max().date()
                    st.sidebar.write(f"Date Range: {min_d} → {max_d}")
            except Exception:
                pass

        # Main content
        self.render_header()
        self.render_metrics()

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Overview", "Map", "Data", "Upload"]
        )

        with tab1:
            self.render_crime_type_chart()
            st.markdown("---")
            self.render_time_analysis()

            # Recent trends
            st.subheader("Recent Risk Trends")
            col1, col2 = st.columns(2)

            with col1:
                if (
                    "Is_Weapon" in self.df.columns
                    and "Month" in self.df.columns
                ):
                    weapon_trend = (
                        self.df.groupby("Month")["Is_Weapon"]
                        .mean()
                        .sort_index()
                    )
                    if not weapon_trend.empty:
                        st.line_chart(weapon_trend)
                        st.caption("Monthly weapon involvement rate")

            with col2:
                if (
                    "Is_Violent" in self.df.columns
                    and "Month" in self.df.columns
                ):
                    violent_trend = (
                        self.df.groupby("Month")["Is_Violent"]
                        .mean()
                        .sort_index()
                    )
                    if not violent_trend.empty:
                        st.line_chart(violent_trend)
                        st.caption("Monthly violent crime rate")

        with tab2:
            self.render_map()

        with tab3:
            self.render_data_preview()

        with tab4:
            self.render_upload_section()

        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; padding: 1rem;'>"
            "Crime Pattern Analysis System • Powered by your processed crime dataset"
            "</div>",
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------
# Run application
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        dashboard = CrimeDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please verify your processed dataset and try again.")
