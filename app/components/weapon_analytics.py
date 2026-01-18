import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


class WeaponAnalyticsEngine:
    """
    Advanced analytics engine focused on weapon-based crime patterns.
    Designed to adapt to the actual dataset columns without generating
    random or synthetic weapon information.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate_and_prepare()

    def _validate_and_prepare(self):
        """
        Clean and align columns with the actual dataset.

        This implementation is data-driven:
        - Uses existing weapon columns (e.g. 'Weapon Desc', 'Weapon Used Cd')
        - Derives flags and temporal fields deterministically from the data
        - Does NOT create random weapon types or random values
        """
        # --- Map weapon columns ---

        # Prefer an existing 'Weapon Type'; otherwise map from 'Weapon Desc'
        if "Weapon Type" not in self.df.columns:
            if "Weapon Desc" in self.df.columns:
                self.df["Weapon Type"] = (
                    self.df["Weapon Desc"]
                    .fillna("Unknown")
                    .astype(str)
                    .str.strip()
                    .replace("", "Unknown")
                )
            else:
                # Fallback: no weapon info at all -> mark Unknown
                self.df["Weapon Type"] = "Unknown"
        else:
            self.df["Weapon Type"] = (
                self.df["Weapon Type"]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
            )

        # --- Weapon used flag ---
        # 1) If a boolean / yes-no column exists, use it.
        # 2) Else if 'Weapon Used Cd' exists, treat 0 or NaN as "no/unknown", others as "weapon present".
        # 3) Else derive from 'Weapon Type' text.
        if "Weapon Used" in self.df.columns:
            self.df["Weapon Used Flag"] = (
                self.df["Weapon Used"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["yes", "y", "true", "1"])
            )
        elif "Weapon Used Cd" in self.df.columns:
            self.df["Weapon Used Flag"] = (
                self.df["Weapon Used Cd"]
                    .fillna(0)
                    .astype(float)
                    .ne(0.0)
            )
        else:
            # Derive from weapon type text: treat 'Unknown', 'No Weapon' style as False
            no_weapon_terms = {"unknown", "none", "no weapon", "unarmed"}
            self.df["Weapon Used Flag"] = ~(
                self.df["Weapon Type"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(no_weapon_terms)
            )

        # --- Date / time handling ---
        if "DateTime" in self.df.columns:
            self.df["DateTime"] = pd.to_datetime(self.df["DateTime"], errors="coerce")
        else:
            # Try to build from typical crime dataset columns (like LA crime data)
            dt = None
            if "DATE OCC" in self.df.columns and "TIME OCC" in self.df.columns:
                dt = pd.to_datetime(
                    self.df["DATE OCC"].astype(str).str.strip() + " " +
                    self.df["TIME OCC"].astype(str).str.strip(),
                    errors="coerce"
                )
            elif "DATE OCC" in self.df.columns:
                dt = pd.to_datetime(self.df["DATE OCC"], errors="coerce")
            elif "Date" in self.df.columns:
                dt = pd.to_datetime(self.df["Date"], errors="coerce")

            if dt is not None:
                self.df["DateTime"] = dt
            else:
                # Last resort: constant timestamp, but deterministic (not random)
                self.df["DateTime"] = pd.to_datetime("2000-01-01")

        # Derive basic temporal components (no randomness)
        if "Hour" not in self.df.columns:
            self.df["Hour"] = self.df["DateTime"].dt.hour
        if "Month" not in self.df.columns:
            self.df["Month"] = self.df["DateTime"].dt.month
        if "Year" not in self.df.columns:
            self.df["Year"] = self.df["DateTime"].dt.year

        # --- Crime type ---
        if "Crime Type" not in self.df.columns:
            # Map from common descriptive column if available
            if "Crm Cd Desc" in self.df.columns:
                self.df["Crime Type"] = self.df["Crm Cd Desc"].astype(str)
            else:
                self.df["Crime Type"] = "Unknown"

        # --- Severity score ---
        # Use existing 'Severity Score' if present; otherwise, keep it absent.
        # Severity-based stats will be skipped when this is missing.
        if "Severity Score" in self.df.columns:
            self.df["Severity Score"] = pd.to_numeric(
                self.df["Severity Score"], errors="coerce"
            )

        # --- Geographic / area naming ---
        # State: if not present, try from AREA NAME (for LA dataset) or mark as Unknown.
        if "State" not in self.df.columns:
            if "AREA NAME" in self.df.columns:
                # Treat LAPD "AREA NAME" as a pseudo-state/region for analytics
                self.df["State"] = self.df["AREA NAME"].astype(str)
            else:
                self.df["State"] = "Unknown"

        # City: optional; if absent, fill with "Unknown"
        if "City" not in self.df.columns:
            self.df["City"] = "Unknown"

        # Area Type: if missing, mark as Unknown (no random labels)
        if "Area Type" not in self.df.columns:
            self.df["Area Type"] = "Unknown"

        # Helper columns for timelines
        self.df["Date"] = self.df["DateTime"].dt.date
        self.df["YearMonth"] = self.df["DateTime"].dt.to_period("M").astype(str)

    # --------- HIGH LEVEL STATS ---------

    def get_overall_weapon_stats(self):
        """Return global stats related to weapon usage (fully data-driven)."""
        total_crimes = len(self.df)
        if total_crimes == 0:
            return {
                "total_crimes": 0,
                "with_weapon": 0,
                "without_weapon": 0,
                "weapon_usage_pct": 0.0,
                "avg_severity_with_weapon": None,
                "avg_severity_without_weapon": None,
                "top_weapons": {}
            }

        with_weapon = int(self.df["Weapon Used Flag"].sum())
        without_weapon = int(total_crimes - with_weapon)
        weapon_usage_pct = round(with_weapon / total_crimes * 100, 1)

        # Severity comparison (only if severity exists)
        if "Severity Score" in self.df.columns:
            with_weapon_severity = self.df.loc[
                self.df["Weapon Used Flag"], "Severity Score"
            ].mean()
            without_weapon_severity = self.df.loc[
                ~self.df["Weapon Used Flag"], "Severity Score"
            ].mean()

            with_weapon_severity = (
                None if np.isnan(with_weapon_severity)
                else round(with_weapon_severity, 2)
            )
            without_weapon_severity = (
                None if np.isnan(without_weapon_severity)
                else round(without_weapon_severity, 2)
            )
        else:
            with_weapon_severity = None
            without_weapon_severity = None

        # Top weapons by count
        top_weapons_series = (
            self.df.loc[self.df["Weapon Used Flag"], "Weapon Type"]
            .value_counts()
            .head(10)
        )
        top_weapons = top_weapons_series.to_dict()

        return {
            "total_crimes": total_crimes,
            "with_weapon": with_weapon,
            "without_weapon": without_weapon,
            "weapon_usage_pct": weapon_usage_pct,
            "avg_severity_with_weapon": with_weapon_severity,
            "avg_severity_without_weapon": without_weapon_severity,
            "top_weapons": top_weapons
        }

    # --------- DISTRIBUTIONS & RELATIONSHIPS ---------

    def weapon_type_distribution(self, top_n=15):
        """Frequency of each weapon type for weapon-involved crimes."""
        subset = self.df.loc[self.df["Weapon Used Flag"], "Weapon Type"]
        if subset.empty:
            return pd.Series(dtype=int)
        weapon_counts = subset.value_counts().head(top_n)
        return weapon_counts

    def severity_by_weapon_type(self, min_count=30):
        """Average severity per weapon type (only when severity exists)."""
        if "Severity Score" not in self.df.columns:
            return pd.DataFrame()

        subset = self.df.loc[self.df["Weapon Used Flag"]].copy()
        if subset.empty:
            return pd.DataFrame()

        grouped = (
            subset
            .groupby("Weapon Type")
            .agg(
                count=("Weapon Type", "size"),
                avg_severity=("Severity Score", "mean"),
                max_severity=("Severity Score", "max")
            )
        )
        grouped = grouped[grouped["count"] >= min_count]
        grouped = grouped.sort_values("avg_severity", ascending=False)
        return grouped

    def crime_type_vs_weapon_matrix(self, min_total=20):
        """Crosstab: Crime Type vs Weapon Type (row-normalized)."""
        subset = self.df.loc[self.df["Weapon Used Flag"]]

        if subset.empty:
            return pd.DataFrame()

        matrix = pd.crosstab(subset["Crime Type"], subset["Weapon Type"])
        matrix = matrix.loc[matrix.sum(axis=1) >= min_total]

        if matrix.empty:
            return pd.DataFrame()

        row_sums = matrix.sum(axis=1)
        matrix_normalized = matrix.div(row_sums, axis=0)
        return matrix_normalized

    def area_type_vs_weapon_matrix(self, min_total=10):
        """Crosstab: Area Type vs Weapon Type (row-normalized)."""
        subset = self.df.loc[self.df["Weapon Used Flag"]]

        if subset.empty or "Area Type" not in subset.columns:
            return pd.DataFrame()

        matrix = pd.crosstab(subset["Area Type"], subset["Weapon Type"])
        matrix = matrix.loc[matrix.sum(axis=1) >= min_total]

        if matrix.empty:
            return pd.DataFrame()

        row_sums = matrix.sum(axis=1)
        matrix_normalized = matrix.div(row_sums, axis=0)
        return matrix_normalized

    def hourly_weapon_pattern(self):
        """Weapon-involved crimes by hour of day."""
        subset = self.df.loc[self.df["Weapon Used Flag"]].copy()
        if subset.empty or "Hour" not in subset.columns:
            return pd.Series(dtype=int)

        hourly_counts = subset["Hour"].value_counts().sort_index()
        return hourly_counts

    def weapon_usage_trend_monthly(self):
        """Trend of weapon-based crimes over YearMonth."""
        subset = self.df.copy()
        subset["WeaponFlag"] = subset["Weapon Used Flag"].astype(int)

        monthly = subset.groupby("YearMonth").agg(
            total_crimes=("WeaponFlag", "size"),
            weapon_crimes=("WeaponFlag", "sum"),
        ).reset_index()

        if "Severity Score" in subset.columns:
            monthly["avg_severity"] = (
                subset.groupby("YearMonth")["Severity Score"].mean().values
            )
        else:
            monthly["avg_severity"] = np.nan

        monthly["weapon_usage_pct"] = (
            monthly["weapon_crimes"] / monthly["total_crimes"].replace(0, np.nan) * 100
        ).fillna(0.0)

        return monthly

    def top_weapons_by_state(self, top_states=10):
        """For each top 'State' (or area), find most common weapon."""
        subset = self.df.loc[self.df["Weapon Used Flag"]].copy()
        if subset.empty or "State" not in subset.columns:
            return pd.DataFrame()

        state_counts = subset["State"].value_counts().head(top_states).index.tolist()

        rows = []
        for state in state_counts:
            state_df = subset[subset["State"] == state]
            if state_df.empty:
                continue
            weapon_counts = state_df["Weapon Type"].value_counts()
            top_weapon = weapon_counts.idxmax()
            weapon_count = weapon_counts.max()
            if "Severity Score" in state_df.columns:
                avg_severity = state_df["Severity Score"].mean()
                avg_severity = round(float(avg_severity), 2)
            else:
                avg_severity = None

            rows.append({
                "State": state,
                "Top Weapon": top_weapon,
                "Weapon Crimes": int(weapon_count),
                "Total Weapon Crimes in State": int(len(state_df)),
                "Avg Severity (Weapon Crimes)": avg_severity
            })

        return pd.DataFrame(rows)

    # --------- VISUALIZATIONS (PLOTLY FIGURES) ---------

    def fig_weapon_type_bar(self, top_n=15):
        counts = self.weapon_type_distribution(top_n)
        if counts.empty:
            return None

        fig = px.bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            title="<b>Top Weapon Types Used in Crimes</b>",
            labels={"x": "Number of Crimes", "y": "Weapon Type"},
            color=counts.values,
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=400)
        return fig

    def fig_weapon_usage_pie(self):
        stats = self.get_overall_weapon_stats()
        if stats["total_crimes"] == 0:
            return None

        values = [stats["with_weapon"], stats["without_weapon"]]
        labels = ["With Weapon", "Without Weapon"]

        fig = px.pie(
            values=values,
            names=labels,
            title="<b>Weapon vs Non-Weapon Crimes</b>",
            hole=0.4
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=350)
        return fig

    def fig_severity_by_weapon(self, min_count=30):
        data = self.severity_by_weapon_type(min_count)
        if data.empty:
            return None

        df_plot = data.reset_index().rename(columns={"index": "Weapon Type"})
        fig = px.bar(
            df_plot,
            x="avg_severity",
            y="Weapon Type",
            orientation="h",
            title="<b>Average Severity by Weapon Type</b>",
            labels={"avg_severity": "Average Severity Score", "Weapon Type": "Weapon Type"},
            color="avg_severity",
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=450)
        return fig

    def fig_hourly_pattern(self):
        hourly = self.hourly_weapon_pattern()
        if hourly.empty:
            return None

        fig = px.line(
            x=hourly.index,
            y=hourly.values,
            markers=True,
            title="<b>Weapon-Based Crimes by Hour of Day</b>",
            labels={"x": "Hour of Day", "y": "Number of Crimes"}
        )
        fig.update_traces(line=dict(width=3))
        fig.update_layout(height=350, xaxis=dict(dtick=1))
        return fig

    def fig_weapon_usage_trend(self):
        monthly = self.weapon_usage_trend_monthly()
        if monthly.empty:
            return None

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["YearMonth"],
            y=monthly["weapon_crimes"],
            name="Weapon Crimes"
        ))
        fig.add_trace(go.Scatter(
            x=monthly["YearMonth"],
            y=monthly["weapon_usage_pct"],
            name="Weapon Usage (%)",
            yaxis="y2",
            mode="lines+markers"
        ))

        fig.update_layout(
            title="<b>Weapon Usage Trend Over Time</b>",
            xaxis_title="Year-Month",
            yaxis=dict(title="Weapon Crimes"),
            yaxis2=dict(
                title="Weapon Usage (%)",
                overlaying="y",
                side="right"
            ),
            height=450
        )
        return fig

    def fig_crime_vs_weapon_heatmap(self, min_total=20):
        matrix = self.crime_type_vs_weapon_matrix(min_total)
        if matrix.empty:
            return None

        fig = px.imshow(
            matrix,
            aspect="auto",
            labels=dict(x="Weapon Type", y="Crime Type", color="Probability"),
            title="<b>Crime Type vs Weapon Type (Normalized)</b>",
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=500)
        return fig

    def fig_area_vs_weapon_heatmap(self, min_total=10):
        matrix = self.area_type_vs_weapon_matrix(min_total)
        if matrix.empty:
            return None

        fig = px.imshow(
            matrix,
            aspect="auto",
            labels=dict(x="Weapon Type", y="Area Type", color="Probability"),
            title="<b>Area Type vs Weapon Type (Normalized)</b>",
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=450)
        return fig

    def fig_top_weapons_by_state(self, top_states=10):
        df_state = self.top_weapons_by_state(top_states)
        if df_state.empty:
            return None

        fig = px.bar(
            df_state,
            x="Weapon Crimes",
            y="State",
            color="Top Weapon",
            orientation="h",
            title="<b>Top Weapons by State</b>",
            labels={"Weapon Crimes": "Number of Crimes", "State": "State", "Top Weapon": "Weapon"}
        )
        fig.update_layout(height=450)
        return fig
