# app/components/analytics_engine.py
"""
Robust analytics engine adapted for the uploaded crime dataset.

Exports:
 - ensure_canonical_columns(df) -> DataFrame
 - get_available_options(df) -> dict
 - CrimeAnalyticsEngine(df) -> class with `.insights`
 - create_advanced_analytics_dashboard(engine) -> dict of visualizations + raw_insights
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# ----------------------------
# Utility helpers
# ----------------------------
def _first_existing_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_hour_value(t):
    """Parse various TIME OCC formats into integer hour (0-23) or np.nan"""
    if pd.isna(t):
        return np.nan
    try:
        # numeric like 1300, 900, 0
        if isinstance(t, (int, float, np.integer, np.floating)):
            t_int = int(t)
            hour = t_int // 100
            if 0 <= hour <= 23:
                return int(hour)
        s = str(t).strip()
        # '1300' string
        if s.isdigit():
            s = s.zfill(4)
            h = int(s[:2])
            if 0 <= h <= 23:
                return h
        # '13:00' or '1:00 PM'
        for fmt in ("%H:%M", "%I:%M %p", "%H%M", "%I%M%p", "%H.%M"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.hour
            except Exception:
                continue
        # fallback: leading digits
        import re

        m = re.match(r"^\s*(\d{1,2})", s)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return h
    except Exception:
        return np.nan
    return np.nan


# ----------------------------
# Canonicalization & Options
# ----------------------------
def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add or normalize canonical columns used across the app and engine.
    Returns a copy of df with canonical columns set:
      - 'FIR Number', 'Crime Type', 'Crime Code', 'City', 'Location',
        'Latitude', 'Longitude', 'Date_parsed', 'Hour', 'DateTime',
        'Year', 'Month', 'Month_Name', 'Day_of_Week_Num', 'Day_of_Week_Name',
        'Is_Weekend', 'Time_Category', 'Part of Day', 'Severity Score', 'State'
    """
    df = df.copy()

    # 1) FIR Number
    id_col = _first_existing_column(df, ["DR_NO", "FIR Number", "FIR_NO", "id"])
    if id_col:
        df["FIR Number"] = df[id_col]
    else:
        df["FIR Number"] = np.arange(1, len(df) + 1)

    # 2) Crime Type and Code
    crime_desc = _first_existing_column(
        df,
        [
            "Crm Cd Desc",
            "Crm_Cd_Desc",
            "CrmCdDesc",
            "Crime Type",
            "crime_type",
            "Crm Cd Desc ",
        ],
    )
    crime_code = _first_existing_column(
        df, ["Crm Cd", "CrmCd", "CRIME CODE", "Crime Code"]
    )

    # If dataset already has a clean Crime Type, keep it
    if "Crime Type" in df.columns:
        df["Crime Type"] = df["Crime Type"].astype(str).str.strip()
    else:
        if crime_desc:
            df["Crime Type"] = df[crime_desc].astype(str).str.strip()
        elif crime_code:
            df["Crime Type"] = df[crime_code].astype(str)
        else:
            df["Crime Type"] = "Unknown"

    if crime_code:
        df["Crime Code"] = df[crime_code]

    # 3) City / Location
    # If dataset already has City, keep it and just normalize
    if "City" in df.columns:
        df["City"] = df["City"].astype(str).str.strip().replace("", "Unknown")
    else:
        area_col = _first_existing_column(
            df, ["AREA NAME", "Area Name", "AREA", "Area"]
        )
        loc_col = _first_existing_column(df, ["LOCATION", "Location", "LOC"])
        if area_col:
            df["City"] = df[area_col].astype(str).str.strip()
        elif loc_col:
            df["City"] = (
                df[loc_col]
                .astype(str)
                .str.split(",")
                .str[-1]
                .str.strip()
                .replace("", "Unknown")
            )
        else:
            df["City"] = "Unknown"

    # Location: if already present, keep; else derive
    if "Location" in df.columns:
        df["Location"] = df["Location"].astype(str)
    else:
        loc_col = _first_existing_column(df, ["LOCATION", "Location", "LOC"])
        if loc_col:
            df["Location"] = df[loc_col].astype(str)
        else:
            df["Location"] = df["City"].astype(str)

    # 4) Latitude / Longitude
    lat_col = _first_existing_column(df, ["LAT", "Latitude", "lat"])
    lon_col = _first_existing_column(df, ["LON", "Longitude", "lon", "LONG"])
    if lat_col:
        df["Latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    else:
        df["Latitude"] = pd.to_numeric(df.get("Latitude", np.nan), errors="coerce")

    if lon_col:
        df["Longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        df["Longitude"] = pd.to_numeric(df.get("Longitude", np.nan), errors="coerce")

    # 5) Date parsing
    date_col = _first_existing_column(
        df,
        [
            "DATE OCC",
            "DATE_OCC",
            "Date Occurred",
            "Date Rptd",
            "DATE RPTD",
            "occ_date",
            "Date",
            "date",
        ],
    )
    if date_col:
        df["Date_parsed"] = pd.to_datetime(
            df[date_col], errors="coerce", dayfirst=False
        )
    else:
        # flexible search for a column containing 'date' but not 'time'
        date_col2 = None
        for c in df.columns:
            if "date" in c.lower() and "time" not in c.lower():
                date_col2 = c
                break
        if date_col2:
            df["Date_parsed"] = pd.to_datetime(
                df[date_col2], errors="coerce", dayfirst=False
            )
        else:
            df["Date_parsed"] = pd.NaT

    # 6) Time / Hour
    time_col = _first_existing_column(
        df,
        [
            "TIME OCC",
            "TIME_OCC",
            "TIME_OCCURRED",
            "Time Occ",
            "TIME",
            "time",
        ],
    )
    if time_col:
        df["Hour"] = df[time_col].apply(_parse_hour_value)
    else:
        # try to extract from DateTime column if present
        if "DateTime" in df.columns:
            try:
                df["Hour"] = pd.to_datetime(
                    df["DateTime"], errors="coerce"
                ).dt.hour
            except Exception:
                df["Hour"] = np.nan
        else:
            df["Hour"] = np.nan

    # 7) Reconstruct DateTime where possible
    def _build_dt(row):
        d = row.get("Date_parsed", pd.NaT)
        h = row.get("Hour", np.nan)
        if pd.isna(d):
            return pd.NaT
        if pd.isna(h):
            return pd.Timestamp(d)
        try:
            return pd.Timestamp(d) + pd.Timedelta(hours=int(h))
        except Exception:
            return pd.Timestamp(d)

    if "DateTime" not in df.columns:
        df["DateTime"] = df.apply(_build_dt, axis=1)
    else:
        # ensure proper datetime
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

    # 8) Year, Month, Day names
    if "occ_year" in df.columns:
        try:
            df["Year"] = pd.to_numeric(
                df["occ_year"], errors="coerce"
            ).astype("Int64")
        except Exception:
            df["Year"] = pd.to_datetime(
                df["DateTime"], errors="coerce"
            ).dt.year
    else:
        df["Year"] = pd.to_datetime(
            df["DateTime"], errors="coerce"
        ).dt.year.astype("Int64")

    if "occ_month" in df.columns:
        try:
            df["Month"] = pd.to_datetime(
                df["occ_month"], format="%b", errors="coerce"
            ).dt.month
            df.loc[df["Month"].isna(), "Month"] = pd.to_datetime(
                df.loc[df["Month"].isna(), "occ_month"], errors="coerce"
            ).dt.month
        except Exception:
            df["Month"] = pd.to_numeric(df["occ_month"], errors="coerce")
        df["Month"] = (
            df["Month"]
            .fillna(pd.to_datetime(df["DateTime"], errors="coerce").dt.month)
            .astype("Int64")
        )
    else:
        df["Month"] = pd.to_datetime(
            df["DateTime"], errors="coerce"
        ).dt.month.astype("Int64")

    df["Month_Name"] = pd.to_datetime(
        df["DateTime"], errors="coerce"
    ).dt.month_name()
    df["Day_of_Week_Num"] = pd.to_datetime(
        df["DateTime"], errors="coerce"
    ).dt.dayofweek
    df["Day_of_Week_Name"] = pd.to_datetime(
        df["DateTime"], errors="coerce"
    ).dt.day_name()
    df["Is_Weekend"] = (
        df["Day_of_Week_Num"]
        .apply(lambda x: 1 if x in (5, 6) else 0)
        .fillna(0)
        .astype(int)
    )

    # 9) Time_Category and Part of Day
    def _time_category(h):
        try:
            if pd.isna(h):
                return "Unknown"
            h = int(h)
            if 0 <= h <= 5:
                return "Night"
            if 6 <= h <= 11:
                return "Morning"
            if 12 <= h <= 17:
                return "Afternoon"
            if 18 <= h <= 23:
                return "Evening"
        except Exception:
            return "Unknown"
        return "Unknown"

    if "Time_Category" not in df.columns:
        df["Time_Category"] = df["Hour"].apply(_time_category).astype(str)
    else:
        df["Time_Category"] = df["Time_Category"].fillna("Unknown").astype(str)

    if "Part of Day" not in df.columns:
        def _pod(h):
            try:
                if pd.isna(h):
                    return "Unknown"
                return "AM" if int(h) < 12 else "PM"
            except Exception:
                return "Unknown"

        df["Part of Day"] = df["Hour"].apply(_pod).astype(str)
    else:
        df["Part of Day"] = df["Part of Day"].fillna("Unknown").astype(str)

    # 10) Severity Score fallback (keep dataset values if present)
    if "Severity Score" not in df.columns:
        if "Part 1-2" in df.columns:
            df["Severity Score"] = pd.to_numeric(
                df["Part 1-2"], errors="coerce"
            ).fillna(1.0)
        else:
            df["Severity Score"] = 1.0
    else:
        df["Severity Score"] = pd.to_numeric(
            df["Severity Score"], errors="coerce"
        ).fillna(1.0)

    # 11) State column always exists
    if "State" not in df.columns:
        df["State"] = "Unknown"
    else:
        df["State"] = df["State"].astype(str).fillna("Unknown")

    # Ensure types
    df["Crime Type"] = df["Crime Type"].fillna("Unknown").astype(str)
    df["City"] = df["City"].fillna("Unknown").astype(str)
    df["Location"] = df["Location"].fillna(df["City"]).astype(str)
    df["Time_Category"] = df["Time_Category"].fillna("Unknown").astype(str)
    df["Part of Day"] = df["Part of Day"].fillna("Unknown").astype(str)

    # Normalize Latitude/Longitude numeric types
    df["Latitude"] = pd.to_numeric(
        df.get("Latitude", pd.Series(np.nan)), errors="coerce"
    )
    df["Longitude"] = pd.to_numeric(
        df.get("Longitude", pd.Series(np.nan)), errors="coerce"
    )

    return df


def get_available_options(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return lists for sidebar selectors built from the DataFrame:
      - cities, years, months, time_categories, part_of_day, crime_types
    """
    if "Crime Type" not in df.columns or "City" not in df.columns:
        df = ensure_canonical_columns(df)

    def _vals(col):
        if col not in df.columns:
            return []
        vals = df[col].dropna().unique().tolist()
        try:
            return sorted(vals)
        except Exception:
            return vals

    out = {
        "cities": _vals("City"),
        "years": sorted(
            [
                int(v)
                for v in set(
                    df.get("Year", pd.Series([], dtype="Int64"))
                    .dropna()
                    .unique()
                )
            ]
        )
        if "Year" in df.columns
        else [],
        "months": sorted(
            [
                int(v)
                for v in set(
                    df.get("Month", pd.Series([], dtype="Int64"))
                    .dropna()
                    .unique()
                )
            ]
        )
        if "Month" in df.columns
        else [],
        "time_categories": _vals("Time_Category"),
        "part_of_day": _vals("Part of Day"),
        "crime_types": _vals("Crime Type"),
    }
    return out


# --------------------------
# CrimeAnalyticsEngine
# --------------------------
class CrimeAnalyticsEngine:
    """
    Analytics engine adapted to the uploaded dataset columns.

    Usage:
      engine = CrimeAnalyticsEngine(df)
      insights = engine.insights
      dashboard = create_advanced_analytics_dashboard(engine)
    """

    def __init__(self, df: pd.DataFrame):
        # canonicalize and keep a copy
        self.df = ensure_canonical_columns(df)
        self.insights: Dict[str, Any] = {}
        self.generate_comprehensive_analytics()

    def calculate_monthly_trend(self):
        monthly_counts = (
            self.df["Month"].dropna().astype(int).value_counts().sort_index()
        )
        if len(monthly_counts) > 1:
            x = np.arange(len(monthly_counts))
            y = monthly_counts.values
            try:
                trend = np.polyfit(x, y, 1)[0]
                return "Increasing" if trend > 0 else "Decreasing" if trend < 0 else "Stable"
            except Exception:
                return "Stable"
        return "Stable"

    def generate_comprehensive_analytics(self):
        self.insights = {
            "temporal_analysis": self.analyze_temporal_patterns(),
            "geospatial_analysis": self.analyze_geospatial_patterns(),
            "crime_type_analysis": self.analyze_crime_types(),
            "weapon_analysis": self.analyze_weapon_patterns(),
            "seasonal_analysis": self.analyze_seasonal_trends(),
            "hotspot_evolution": self.analyze_hotspot_evolution(),
            "predictive_trends": self.analyze_predictive_trends(),
            "correlation_analysis": self.analyze_correlations(),
        }

    # --- Temporal ---
    def analyze_temporal_patterns(self):
        patterns = {}
        hour_series = (
            self.df["Hour"].dropna().astype(int)
            if "Hour" in self.df.columns
            else pd.Series(dtype=int)
        )
        hourly_counts = hour_series.value_counts().sort_index()
        patterns["hourly"] = {
            "distribution": hourly_counts,
            "peak_hours": hourly_counts.nlargest(5).to_dict(),
            "low_hours": hourly_counts.nsmallest(5).to_dict(),
        }

        if "Day_of_Week_Num" in self.df.columns:
            daily_counts = self.df["Day_of_Week_Num"].value_counts().sort_index()
            day_names = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            distrib = {day_names[i]: int(daily_counts.get(i, 0)) for i in range(7)}
            patterns["daily"] = {
                "distribution": distrib,
                "busiest_day": max(distrib, key=distrib.get),
                "quietest_day": min(distrib, key=distrib.get),
            }

        monthly_counts = (
            self.df["Month"].dropna().astype(int).value_counts().sort_index()
            if "Month" in self.df.columns
            else pd.Series(dtype=int)
        )
        patterns["monthly"] = {
            "distribution": monthly_counts,
            "peak_month": int(monthly_counts.idxmax())
            if len(monthly_counts) > 0
            else None,
            "trend": self.calculate_monthly_trend(),
        }

        if "Is_Weekend" in self.df.columns:
            weekend_stats = (
                self.df.groupby("Is_Weekend")
                .agg({"FIR Number": "count", "Severity Score": "mean"})
                .rename(columns={"FIR Number": "count"})
            )
            weekend_count = (
                int(weekend_stats.loc[1, "count"])
                if 1 in weekend_stats.index
                else 0
            )
            weekday_count = (
                int(weekend_stats.loc[0, "count"])
                if 0 in weekend_stats.index
                else 0
            )
            patterns["weekend_analysis"] = {
                "weekend_crime_rate": weekend_count / len(self.df)
                if len(self.df) > 0
                else 0,
                "weekday_crime_rate": weekday_count / len(self.df)
                if len(self.df) > 0
                else 0,
                "weekend_severity": float(
                    weekend_stats.loc[1, "Severity Score"]
                )
                if 1 in weekend_stats.index
                else np.nan,
                "weekday_severity": float(
                    weekend_stats.loc[0, "Severity Score"]
                )
                if 0 in weekend_stats.index
                else np.nan,
            }

        return patterns

    # --- Geospatial ---
    def identify_geographic_clusters(self):
        if "Latitude" not in self.df.columns or "Longitude" not in self.df.columns:
            return pd.DataFrame()
        self.df["Lat_Cluster"] = self.df["Latitude"].round(1)
        self.df["Lon_Cluster"] = self.df["Longitude"].round(1)
        clusters = (
            self.df.groupby(["Lat_Cluster", "Lon_Cluster"])
            .agg(
                {
                    "FIR Number": "count",
                    "Severity Score": "mean",
                    "State": lambda x: x.mode().iloc[0]
                    if not x.mode().empty
                    else "Unknown",
                    "City": lambda x: x.mode().iloc[0]
                    if not x.mode().empty
                    else "Unknown",
                }
            )
            .rename(
                columns={
                    "FIR Number": "Crime_Count",
                    "Severity Score": "Avg_Severity",
                }
            )
            .reset_index()
        )
        return clusters.sort_values("Crime_Count", ascending=False).head(10)

    def calculate_risk_density(self):
        if "Lat_Cluster" not in self.df.columns or "Lon_Cluster" not in self.df.columns:
            _ = self.identify_geographic_clusters()
        if "Lat_Cluster" in self.df.columns and "Lon_Cluster" in self.df.columns:
            risk_density = (
                self.df.groupby(["Lat_Cluster", "Lon_Cluster"])
                .agg({"FIR Number": "count", "Severity Score": "mean"})
                .rename(columns={"FIR Number": "Crime_Count"})
            )
            if len(risk_density) > 0:
                crime_max = risk_density["Crime_Count"].max()
                severity_max = (
                    risk_density["Severity Score"].max()
                    if "Severity Score" in risk_density.columns
                    else 1
                )
                if crime_max > 0 and severity_max > 0:
                    risk_density["Risk_Score"] = (
                        (risk_density["Crime_Count"] / crime_max) * 0.7
                        + (risk_density["Severity Score"] / severity_max) * 0.3
                    )
                else:
                    risk_density["Risk_Score"] = 0
                return (
                    risk_density.sort_values("Risk_Score", ascending=False)
                    .head(15)
                    .reset_index()
                )
        return pd.DataFrame()

    def analyze_geospatial_patterns(self):
        """Advanced geospatial analysis; guarded against missing State/coords"""
        spatial: Dict[str, Any] = {}

        # State-level analysis
        if "State" in self.df.columns and not self.df["State"].dropna().empty:
            try:
                state_stats = (
                    self.df.groupby("State")
                    .agg(
                        {
                            "FIR Number": "count",
                            "Severity Score": ["mean", "max", "min"],
                            "Latitude": "first",
                            "Longitude": "first",
                        }
                    )
                    .round(2)
                )
                if not state_stats.empty:
                    state_stats.columns = [
                        "Crime_Count",
                        "Avg_Severity",
                        "Max_Severity",
                        "Min_Severity",
                        "Latitude",
                        "Longitude",
                    ]
                    spatial["state_analysis"] = state_stats.sort_values(
                        "Crime_Count", ascending=False
                    ).head(10)
                else:
                    spatial["state_analysis"] = pd.DataFrame()
            except Exception:
                spatial["state_analysis"] = pd.DataFrame()
        else:
            spatial["state_analysis"] = pd.DataFrame()

        # City-level hotspots
        if "City" in self.df.columns and not self.df["City"].dropna().empty:
            try:
                city_stats = (
                    self.df.groupby("City")
                    .agg({"FIR Number": "count", "Severity Score": "mean"})
                    .rename(columns={"FIR Number": "count"})
                )
                spatial["city_hotspots"] = city_stats.sort_values(
                    "count", ascending=False
                ).head(15)
            except Exception:
                spatial["city_hotspots"] = pd.DataFrame()
        else:
            spatial["city_hotspots"] = pd.DataFrame()

        spatial["clusters"] = self.identify_geographic_clusters()
        spatial["risk_density"] = self.calculate_risk_density()
        return spatial

    # --- Crime type analysis ---
    def analyze_crime_types(self):
        crime_analysis: Dict[str, Any] = {}
        crime_dist = (
            self.df["Crime Type"].value_counts()
            if "Crime Type" in self.df.columns
            else pd.Series(dtype=int)
        )
        crime_analysis["distribution"] = crime_dist.head(15)

        if "Crime Type" in self.df.columns:
            severity_by_crime = (
                self.df.groupby("Crime Type")["Severity Score"]
                .agg(["mean", "max", "count"])
                .round(2)
            )
            severity_by_crime = severity_by_crime.rename(
                columns={"mean": "mean", "max": "max", "count": "count"}
            )
            crime_analysis["severity_analysis"] = severity_by_crime.sort_values(
                "mean", ascending=False
            ).head(10)
        else:
            crime_analysis["severity_analysis"] = pd.DataFrame()

        crime_analysis["temporal_patterns"] = (
            self.analyze_crime_type_temporal_patterns()
        )

        if "Weapon Type" in self.df.columns:
            crime_analysis["weapon_correlation"] = pd.crosstab(
                self.df["Crime Type"],
                self.df["Weapon Type"],
                normalize="index",
            )

        if "Area Type" in self.df.columns:
            crime_analysis["area_preferences"] = pd.crosstab(
                self.df["Crime Type"],
                self.df["Area Type"],
                normalize="index",
            )

        return crime_analysis

    def analyze_crime_type_temporal_patterns(self):
        temporal_patterns = {}
        if "Crime Type" not in self.df.columns:
            return temporal_patterns
        top_crimes = self.df["Crime Type"].value_counts().head(5).index
        for crime_type in top_crimes:
            crime_data = self.df[self.df["Crime Type"] == crime_type]
            hourly_pattern = (
                crime_data["Hour"].dropna().astype(int).value_counts().sort_index()
                if "Hour" in crime_data.columns
                else pd.Series(dtype=int)
            )
            temporal_patterns[crime_type] = {
                "hourly_distribution": hourly_pattern.to_dict(),
                "peak_hour": int(hourly_pattern.idxmax())
                if len(hourly_pattern) > 0
                else None,
                "total_crimes": int(len(crime_data)),
            }
        return temporal_patterns

    # --- Weapon patterns ---
    def analyze_weapon_patterns(self):
        weapon_analysis = {}
        if "Weapon Type" in self.df.columns:
            try:
                weapon_analysis["distribution"] = self.df[
                    "Weapon Type"
                ].value_counts()
                weapon_analysis["severity_impact"] = (
                    self.df.groupby("Weapon Type")["Severity Score"]
                    .mean()
                    .sort_values(ascending=False)
                )
                weapon_analysis["crime_associations"] = pd.crosstab(
                    self.df["Weapon Type"], self.df["Crime Type"]
                )
                weapon_analysis["temporal_patterns"] = pd.crosstab(
                    self.df["Weapon Type"], self.df["Hour"]
                )
            except Exception:
                pass
        return weapon_analysis

    # --- Seasonal ---
    def analyze_seasonal_trends(self):
        seasonal = {}
        if "Month" in self.df.columns:
            try:
                monthly_trends = (
                    self.df.groupby("Month")
                    .agg({"FIR Number": "count", "Severity Score": "mean"})
                    .rename(columns={"FIR Number": "count"})
                )
                seasonal["monthly_trends"] = monthly_trends
            except Exception:
                seasonal["monthly_trends"] = pd.DataFrame()
        else:
            seasonal["monthly_trends"] = pd.DataFrame()

        season_mapping = {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
        self.df["Season"] = self.df["Month"].map(season_mapping).fillna(
            "Unknown"
        )
        seasonal_stats = (
            self.df.groupby("Season")
            .agg({"FIR Number": "count", "Severity Score": "mean"})
            .rename(columns={"FIR Number": "count"})
        )
        seasonal["seasonal_analysis"] = seasonal_stats

        if "Year" in self.df.columns and self.df["Year"].nunique() > 1:
            yearly_trends = (
                self.df.groupby("Year")
                .agg({"FIR Number": "count", "Severity Score": "mean"})
                .rename(columns={"FIR Number": "count"})
            )
            seasonal["yearly_trends"] = yearly_trends

        return seasonal

    # --- Hotspot evolution ---
    def analyze_hotspot_evolution(self):
        evolution = {}
        if "DateTime" in self.df.columns and not self.df["DateTime"].isna().all():
            try:
                self.df["YearMonth"] = self.df["DateTime"].dt.to_period("M")
                monthly_hotspots = (
                    self.df.groupby("YearMonth")
                    .agg(
                        {
                            "FIR Number": "count",
                            "State": lambda x: x.mode().iloc[0]
                            if not x.mode().empty
                            else "Unknown",
                            "City": lambda x: x.mode().iloc[0]
                            if not x.mode().empty
                            else "Unknown",
                        }
                    )
                    .rename(columns={"FIR Number": "count"})
                )
                evolution["monthly_evolution"] = monthly_hotspots

                recent_cutoff = self.df["DateTime"].quantile(0.75)
                old_cutoff = self.df["DateTime"].quantile(0.25)
                recent_data = self.df[self.df["DateTime"] > recent_cutoff]
                old_data = self.df[self.df["DateTime"] <= old_cutoff]

                if len(recent_data) > 0 and len(old_data) > 0:
                    recent_crimes = recent_data["Crime Type"].value_counts(
                        normalize=True
                    )
                    old_crimes = old_data["Crime Type"].value_counts(
                        normalize=True
                    )
                    all_crimes = set(recent_crimes.index) | set(
                        old_crimes.index
                    )
                    crime_changes = {}
                    for crime in all_crimes:
                        recent_rate = recent_crimes.get(crime, 0)
                        old_rate = old_crimes.get(crime, 0)
                        if old_rate > 0:
                            crime_changes[crime] = (recent_rate - old_rate) / (
                                old_rate
                            )
                        else:
                            crime_changes[crime] = recent_rate
                    evolution["emerging_crimes"] = (
                        pd.Series(crime_changes)
                        .sort_values(ascending=False)
                        .head(10)
                    )
            except Exception:
                pass
        return evolution

    # --- Predictive trends ---
    def analyze_predictive_trends(self):
        trends = {}
        if "Year" in self.df.columns and self.df["Year"].nunique() > 1:
            yearly_crime_counts = self.df.groupby(["Year", "Crime Type"]).size().unstack(fill_value=0)
            if len(yearly_crime_counts) > 1:
                growth_rates = {}
                for crime_type in yearly_crime_counts.columns:
                    values = yearly_crime_counts[crime_type].values
                    if len(values) > 1 and values[0] > 0:
                        growth_rate = (values[-1] - values[0]) / values[0]
                        growth_rates[crime_type] = growth_rate
                trends["growth_rates"] = (
                    pd.Series(growth_rates)
                    .sort_values(ascending=False)
                    .head(10)
                )

        if "Month" in self.df.columns:
            monthly_counts = self.df.groupby("Month")["FIR Number"].count()
            if len(monthly_counts) > 0:
                monthly_std = monthly_counts.std()
                monthly_mean = monthly_counts.mean()
                trends["seasonal_consistency"] = {
                    "predictability_score": float(
                        1 - (monthly_std / monthly_mean)
                    )
                    if monthly_mean > 0
                    else 0,
                    "most_predictable_month": int(monthly_counts.idxmin())
                    if len(monthly_counts) > 0
                    else None,
                    "least_predictable_month": int(monthly_counts.idxmax())
                    if len(monthly_counts) > 0
                    else None,
                }

        return trends

    # --- Correlations ---
    def analyze_correlations(self):
        correlations = {}
        numeric_columns = ["Severity Score", "Hour", "Month", "Year", "Day_of_Week_Num"]
        available_numeric = [
            col for col in numeric_columns if col in self.df.columns
        ]
        if len(available_numeric) > 1:
            correlations["numeric_correlations"] = self.df[
                available_numeric
            ].corr()
        if "Area Type" in self.df.columns:
            correlations["area_crime_association"] = pd.crosstab(
                self.df["Crime Type"], self.df["Area Type"]
            )
        return correlations


# --------------------------
# Visualization helpers
# --------------------------
def create_temporal_visualizations(
    temporal_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    viz = {}
    hourly_data = temporal_analysis.get("hourly", {}).get(
        "distribution", pd.Series(dtype=int)
    )
    if isinstance(hourly_data, (pd.Series, pd.DataFrame)):
        x = hourly_data.index
        y = hourly_data.values
    else:
        x = list(hourly_data.keys())
        y = list(hourly_data.values)
    fig_hourly = px.line(
        x=x,
        y=y,
        title="<b>Crime Distribution by Hour of Day</b>",
        labels={"x": "Hour of Day", "y": "Number of Crimes"},
        line_shape="spline",
    )
    fig_hourly.update_traces(line=dict(width=3))
    fig_hourly.update_layout(height=400, showlegend=False)
    viz["hourly_trend"] = fig_hourly

    if "daily" in temporal_analysis:
        daily_data = temporal_analysis["daily"]["distribution"]
        fig_daily = px.bar(
            x=list(daily_data.keys()),
            y=list(daily_data.values()),
            title="<b>Crime Distribution by Day of Week</b>",
            labels={"x": "Day of Week", "y": "Number of Crimes"},
        )
        fig_daily.update_layout(height=400, showlegend=False)
        viz["daily_pattern"] = fig_daily

    return viz


def create_geospatial_visualizations(
    geospatial_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    viz = {}
    try:
        state_data = geospatial_analysis.get("state_analysis", pd.DataFrame()).reset_index()
        if not state_data.empty and "Crime_Count" in state_data.columns:
            fig_states = px.bar(
                state_data.head(10),
                x="Crime_Count",
                y="State",
                orientation="h",
                title="<b>Top 10 States by Crime Count</b>",
                color="Crime_Count",
            )
            fig_states.update_layout(height=500)
            viz["state_chart"] = fig_states
    except Exception:
        pass
    if (
        "city_hotspots" in geospatial_analysis
        and not geospatial_analysis["city_hotspots"].empty
    ):
        city_data = geospatial_analysis["city_hotspots"].reset_index().head(10)
        fig_cities = px.bar(
            city_data,
            x="count",
            y="City",
            orientation="h",
            title="<b>Top 10 Cities by Crime Count</b>",
            color="count",
        )
        fig_cities.update_layout(height=400)
        viz["city_ranking"] = fig_cities
    return viz


def create_crime_type_visualizations(
    crime_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    viz = {}
    crime_dist = crime_analysis.get("distribution", pd.Series()).head(10)
    if not crime_dist.empty:
        fig_dist = px.pie(
            names=crime_dist.index,
            values=crime_dist.values,
            title="<b>Top 10 Crime Type Distribution</b>",
            hole=0.4,
        )
        fig_dist.update_layout(height=400)
        viz["crime_distribution"] = fig_dist

    if "severity_analysis" in crime_analysis:
        try:
            severity_data = crime_analysis["severity_analysis"].reset_index().head(
                10
            )
            fig_severity = px.bar(
                severity_data,
                x="mean",
                y="Crime Type",
                orientation="h",
                title="<b>Average Severity by Crime Type</b>",
                color="mean",
            )
            fig_severity.update_layout(height=400)
            viz["severity_analysis"] = fig_severity
        except Exception:
            pass

    return viz


def create_seasonal_visualizations(
    seasonal_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    viz = {}
    monthly_df = seasonal_analysis.get("monthly_trends", pd.DataFrame())
    monthly_data = monthly_df.reset_index() if monthly_df is not None else pd.DataFrame()
    if not monthly_data.empty:
        try:
            fig_monthly = px.line(
                monthly_data,
                x="Month",
                y="count",
                title="<b>Monthly Crime Trends</b>",
                markers=True,
            )
            fig_monthly.update_traces(line=dict(width=3))
            fig_monthly.update_layout(height=400, showlegend=False)
            viz["monthly_trends"] = fig_monthly
        except Exception:
            pass
    if "seasonal_analysis" in seasonal_analysis:
        try:
            seasonal_data = seasonal_analysis["seasonal_analysis"].reset_index()
            fig_seasonal = px.bar(
                seasonal_data,
                x="Season",
                y="count",
                title="<b>Crime Distribution by Season</b>",
                color="count",
            )
            fig_seasonal.update_layout(height=400)
            viz["seasonal_analysis"] = fig_seasonal
        except Exception:
            pass
    return viz


def create_advanced_analytics_dashboard(
    analytics_engine: CrimeAnalyticsEngine,
) -> Dict[str, Any]:
    insights = analytics_engine.insights
    dashboard = {
        "temporal_viz": create_temporal_visualizations(
            insights.get("temporal_analysis", {})
        ),
        "geospatial_viz": create_geospatial_visualizations(
            insights.get("geospatial_analysis", {})
        ),
        "crime_type_viz": create_crime_type_visualizations(
            insights.get("crime_type_analysis", {})
        ),
        "seasonal_viz": create_seasonal_visualizations(
            insights.get("seasonal_analysis", {})
        ),
        "raw_insights": insights,
    }
    return dashboard
