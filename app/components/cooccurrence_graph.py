# app/components/cooccurrence_graph.py
"""
Co-occurrence graph utilities for Crime Pattern Analysis project.

- Canonicalization: detect common raw column names and create canonical columns:
    Crime Type, Crime Code, City, Location, Hour, Time_Category, Part of Day, Year, Month, FIR Number, Latitude, Longitude
- Functions:
    - ensure_canonical_columns(df)
    - get_available_options(df)
    - filter_dataset_for_network(df, ...)
    - compute_cooccurrence_edges(df, grouping_context, min_pair_support)
    - compute_node_stats(df, edges_df)
    - create_network_figure(edges_df, node_df, max_nodes)
    - create_cooccurrence_heatmap(edges_df)
"""

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations


# ----------------------------
# Canonicalization helpers
# ----------------------------
def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_hour_from_value(v) -> Optional[int]:
    try:
        if pd.isna(v):
            return None
        s = str(v).strip()
        # numeric format like 1300 or 900
        if s.isdigit():
            s = s.zfill(4)
            h = int(s[:2])
            if 0 <= h <= 23:
                return h
        # formats like "13:00" or "1:30"
        if ":" in s:
            parts = s.split(":")
            h = int(parts[0])
            if 0 <= h <= 23:
                return h
        # fallback: first two chars if plausible
        if len(s) >= 2 and s[:2].isdigit():
            h = int(s[:2])
            if 0 <= h <= 23:
                return h
    except Exception:
        return None
    return None


def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce canonical columns on the DataFrame in-place (returns df for chaining).
    Uses existing dataset columns (guessed names) to populate canonical names:

    Canonical columns created/populated:
      - Crime Type (string)           <-- Crm Cd Desc, Crm Cd
      - Crime Code (optional)         <-- Crm Cd
      - City                           <-- AREA NAME or Location-derived coarse
      - Location                        <-- LOCATION (raw)
      - FIR Number                      <-- DR_NO or create index
      - Hour (0-23 numeric)            <-- TIME OCC or parsed from DateTime
      - Time_Category (Night/Morning/Afternoon/Evening)
      - Part of Day (AM/PM)
      - Year, Month (numeric)
      - LAT, LON (if present)
    """
    # avoid modifying original reference unexpectedly — operate in place but return df
    # CRIME TYPE
    crime_col = _first_existing_column(df, ["Crm Cd Desc", "Crm Cd", "Crime Type", "crime_type", "Crm_Cd_Desc"])
    if crime_col:
        df["Crime Type"] = df[crime_col].astype(str).str.strip()
    else:
        df["Crime Type"] = "Unknown"

    # CRIME CODE (optional)
    code_col = _first_existing_column(df, ["Crm Cd", "CRIME CODE", "CrmCd"])
    if code_col:
        df["Crime Code"] = df[code_col]

    # CITY
    city_col = _first_existing_column(df, ["AREA NAME", "Area Name", "City", "AREA", "Location"])
    if city_col:
        # If AREA NAME exists it's likely the city; keep as-is
        df["City"] = df[city_col].astype(str).str.strip()
    else:
        df["City"] = "Unknown"

    # LOCATION (raw)
    loc_col = _first_existing_column(df, ["LOCATION", "Location", "LOC"])
    if loc_col:
        df["Location"] = df[loc_col].astype(str)
    else:
        # fallback: use City
        df["Location"] = df.get("City", "Unknown").astype(str)

    # FIR Number / unique id
    id_col = _first_existing_column(df, ["DR_NO", "FIR Number", "FIR_NO", "id"])
    if id_col:
        df["FIR Number"] = df[id_col]
    else:
        df["FIR Number"] = range(1, len(df) + 1)

    # LAT / LON
    lat_col = _first_existing_column(df, ["LAT", "Latitude", "lat"])
    lon_col = _first_existing_column(df, ["LON", "Longitude", "lon", "LONG"])
    if lat_col:
        df["LAT"] = pd.to_numeric(df[lat_col], errors="coerce")
    if lon_col:
        df["LON"] = pd.to_numeric(df[lon_col], errors="coerce")

    # Hour: try TIME OCC or TIME_OCC or derive from DateTime
    time_col = _first_existing_column(df, ["TIME OCC", "TIME_OCC", "Time Occ", "Time", "time_occurred", "TIME"])
    if time_col:
        df["Hour"] = df[time_col].apply(_parse_hour_from_value).astype("Int64")
    # if DateTime exists, derive Hour where missing
    if "Hour" not in df.columns or df["Hour"].isna().all():
        dt_col = _first_existing_column(df, ["DateTime", "date_time", "Date Occurred", "DATE"])
        if dt_col:
            try:
                df["Hour"] = pd.to_datetime(df[dt_col], errors="coerce").dt.hour.astype("Int64")
            except Exception:
                pass

    # Time_Category and Part of Day
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
        if "Hour" in df.columns:
            df["Time_Category"] = df["Hour"].apply(_time_category).astype(str)
        else:
            df["Time_Category"] = "Unknown"

    if "Part of Day" not in df.columns:
        if "Hour" in df.columns:
            df["Part of Day"] = df["Hour"].apply(lambda h: "AM" if (not pd.isna(h) and int(h) < 12) else ("PM" if not pd.isna(h) else "Unknown")).astype(str)
        else:
            df["Part of Day"] = "Unknown"

    # Year/Month: prefer occ_year/occ_month or DateTime
    if "Year" not in df.columns:
        y_col = _first_existing_column(df, ["occ_year", "YEAR", "year"])
        if y_col:
            df["Year"] = pd.to_numeric(df[y_col], errors="coerce").astype("Int64")
        else:
            dt_col = _first_existing_column(df, ["DateTime", "date_time", "DATE"])
            if dt_col:
                df["Year"] = pd.to_datetime(df[dt_col], errors="coerce").dt.year.astype("Int64")

    if "Month" not in df.columns:
        m_col = _first_existing_column(df, ["occ_month", "MONTH", "month"])
        if m_col:
            # try to parse textual months first
            try:
                df["Month"] = pd.to_datetime(df[m_col], format="%b", errors="coerce").dt.month
            except Exception:
                df["Month"] = pd.to_numeric(df[m_col], errors="coerce").astype("Int64")
        else:
            dt_col = _first_existing_column(df, ["DateTime", "date_time", "DATE"])
            if dt_col:
                df["Month"] = pd.to_datetime(df[dt_col], errors="coerce").dt.month.astype("Int64")

    # Ensure types and fill missing
    df["Crime Type"] = df["Crime Type"].fillna("Unknown").astype(str)
    df["City"] = df["City"].fillna("Unknown").astype(str)
    df["Location"] = df["Location"].fillna(df["City"]).astype(str)
    df["Time_Category"] = df.get("Time_Category", pd.Series(["Unknown"] * len(df))).fillna("Unknown").astype(str)
    df["Part of Day"] = df.get("Part of Day", pd.Series(["Unknown"] * len(df))).fillna("Unknown").astype(str)

    return df


# ----------------------------
# Option listing for UI
# ----------------------------
def get_available_options(df: pd.DataFrame) -> Dict[str, List]:
    """
    Return a dict of available option lists for populating sidebar selectors,
    using canonical columns (assumes ensure_canonical_columns has been called).
    Keys: states, cities, years, months, area_types, time_categories, part_of_day
    """
    out = {}
    # safe-get helper
    def _vals(col):
        if col in df.columns:
            vals = df[col].dropna().unique().tolist()
            # sort if string-like
            try:
                vals_sorted = sorted(vals)
            except Exception:
                vals_sorted = vals
            return vals_sorted
        return []

    out["states"] = _vals("State")
    out["cities"] = _vals("City")
    out["years"] = sorted([int(x) for x in _vals("Year") if pd.notna(x)]) if "Year" in df.columns else []
    out["months"] = sorted([int(x) for x in _vals("Month") if pd.notna(x)]) if "Month" in df.columns else []
    out["area_types"] = _vals("Area Type") if "Area Type" in df.columns else _vals("AREA TYPE")
    out["time_categories"] = _vals("Time_Category")
    out["part_of_day"] = _vals("Part of Day")
    out["crime_types"] = _vals("Crime Type")
    return out


# ----------------------------
# Filtering function (safe)
# ----------------------------
def filter_dataset_for_network(
    df: pd.DataFrame,
    states: Optional[List] = None,
    cities: Optional[List] = None,
    years: Optional[List] = None,
    months: Optional[List] = None,
    area_types: Optional[List] = None,
    time_categories: Optional[List] = None,
    part_of_day: Optional[List] = None,
) -> pd.DataFrame:
    """
    Filter the dataset based on provided selection lists. Tolerant to missing columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    if states and "State" in data.columns:
        data = data[data["State"].isin(states)]

    if cities and "City" in data.columns:
        data = data[data["City"].isin(cities)]

    if years and "Year" in data.columns:
        data = data[data["Year"].isin(years)]

    if months and "Month" in data.columns:
        data = data[data["Month"].isin(months)]

    if area_types and ("Area Type" in data.columns):
        data = data[data["Area Type"].isin(area_types)]

    if time_categories and "Time_Category" in data.columns:
        data = data[data["Time_Category"].isin(time_categories)]

    if part_of_day and "Part of Day" in data.columns:
        data = data[data["Part of Day"].isin(part_of_day)]

    return data


# ----------------------------
# Grouping config mapping
# ----------------------------
def get_grouping_config(context: str) -> List[str]:
    """
    Map user-selectable grouping contexts to canonical column lists.
    """
    mapping = {
        "City + Time Category": ["City", "Time_Category"],
        "City + Part of Day": ["City", "Part of Day"],
        "City + Date": ["City", "FIR Number"],  # City + Date may require a date column; fallback to city+id (coarse)
        "Area Type + Time Category": ["Area Type", "Time_Category"],
        # additional coarse mappings:
        "City": ["City"],
        "State + City": ["State", "City"],
        "State": ["State"],
    }
    return mapping.get(context, ["City", "Time_Category"])


# ----------------------------
# Compute co-occurrence edges
# ----------------------------
def compute_cooccurrence_edges(
    df: pd.DataFrame,
    grouping_context: str = "City + Time Category",
    min_pair_support: int = 1,
) -> pd.DataFrame:
    """
    Return edges DataFrame with columns: source, target, weight.
    Approach:
      - group by grouping columns (resolved via get_grouping_config)
      - for each group take unique Crime Type set
      - form unordered pairs and count across groups
      - return pairs with count >= min_pair_support
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])

    crime_col = "Crime Type"
    if crime_col not in df.columns:
        return pd.DataFrame(columns=["source", "target", "weight"])

    group_cols = get_grouping_config(grouping_context)
    # keep only those group cols present in df
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        # no valid grouping columns -> return empty
        return pd.DataFrame(columns=["source", "target", "weight"])

    work = df[group_cols + [crime_col]].dropna(subset=[crime_col])

    pair_counts: Dict[Tuple[str, str], int] = {}
    grouped = work.groupby(group_cols, dropna=False)

    for _, group in grouped:
        crime_types = group[crime_col].astype(str).unique()
        if len(crime_types) < 2:
            continue
        crime_types_sorted = sorted(crime_types)
        for a, b in combinations(crime_types_sorted, 2):
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    if not pair_counts:
        return pd.DataFrame(columns=["source", "target", "weight"])

    rows = []
    for (a, b), w in pair_counts.items():
        if w >= (min_pair_support or 1):
            rows.append({"source": a, "target": b, "weight": int(w)})

    edges_df = pd.DataFrame(rows)
    if edges_df.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])

    edges_df = edges_df.sort_values("weight", ascending=False).reset_index(drop=True)
    return edges_df


# ----------------------------
# Compute node stats
# ----------------------------
def compute_node_stats(df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the original df and edges_df, compute per-node stats:
      - crime_type (node id)
      - frequency (how many records of that crime_type in df)
      - degree_strength (sum of incident edge weights)
    """
    if edges_df is None or edges_df.empty:
        return pd.DataFrame(columns=["crime_type", "frequency", "degree_strength"])

    crime_col = "Crime Type"
    freq_series = df[crime_col].astype(str).value_counts()

    strength: Dict[str, float] = {}
    for _, row in edges_df.iterrows():
        s = row["source"]
        t = row["target"]
        w = float(row["weight"])
        strength[s] = strength.get(s, 0.0) + w
        strength[t] = strength.get(t, 0.0) + w

    rows = []
    for crime, strg in strength.items():
        rows.append({
            "crime_type": crime,
            "frequency": int(freq_series.get(crime, 0)),
            "degree_strength": float(strg),
        })

    node_df = pd.DataFrame(rows)
    if node_df.empty:
        return pd.DataFrame(columns=["crime_type", "frequency", "degree_strength"])
    node_df = node_df.sort_values("degree_strength", ascending=False).reset_index(drop=True)
    return node_df


# ----------------------------
# Create network figure (Plotly)
# ----------------------------
def create_network_figure(edges_df: pd.DataFrame, node_df: pd.DataFrame, max_nodes: int = 25) -> go.Figure:
    """
    Build a Plotly figure showing the co-occurrence network using networkx spring layout.
    - Select top nodes by degree_strength (node_df ordered expected)
    - Keep only edges that connect top nodes
    - Visual encoding:
        node size -> frequency (scaled)
        node color -> degree_strength (scaled)
        edge width -> weight scaled
    """
    if edges_df is None or edges_df.empty or node_df is None or node_df.empty:
        return go.Figure()

    # Ensure node_df sorted by degree_strength
    if "degree_strength" in node_df.columns:
        node_df_sorted = node_df.sort_values("degree_strength", ascending=False)
    elif "frequency" in node_df.columns:
        node_df_sorted = node_df.sort_values("frequency", ascending=False)
    else:
        node_df_sorted = node_df.copy()

    top_nodes = node_df_sorted.head(max_nodes)["crime_type"].tolist()
    if not top_nodes:
        return go.Figure()

    edges_filtered = edges_df[edges_df["source"].isin(top_nodes) & edges_df["target"].isin(top_nodes)].copy()
    if edges_filtered.empty:
        return go.Figure()

    # Build graph
    G = nx.Graph()
    # maps
    freq_map = dict(zip(node_df_sorted["crime_type"], node_df_sorted.get("frequency", pd.Series(1, index=node_df_sorted["crime_type"])).tolist()))
    strength_map = dict(zip(node_df_sorted["crime_type"], node_df_sorted.get("degree_strength", pd.Series(1, index=node_df_sorted["crime_type"])).tolist()))

    for n in top_nodes:
        G.add_node(n, frequency=int(freq_map.get(n, 0)), strength=float(strength_map.get(n, 0.0)))

    edges_filtered["weight"] = edges_filtered["weight"].astype(float)
    max_w = float(edges_filtered["weight"].max()) if not edges_filtered["weight"].empty else 1.0

    for _, row in edges_filtered.iterrows():
        G.add_edge(row["source"], row["target"], weight=float(row["weight"]))

    # layout (deterministic seed where available)
    try:
        pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)
    except TypeError:
        pos = nx.spring_layout(G, k=0.6, iterations=50)

    # edge trace
    edge_x = []
    edge_y = []
    edge_widths = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        w = d.get("weight", 1.0)
        # scaled width
        width = 1 + 5 * (w / max_w) if max_w > 0 else 1
        edge_widths.append(width)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # nodes
    node_x = []
    node_y = []
    node_sizes = []
    node_text = []
    node_colors = []

    max_freq = max((freq_map.get(n, 1) for n in top_nodes), default=1)
    max_strength = max((strength_map.get(n, 1) for n in top_nodes), default=1)

    for node in top_nodes:
        x, y = pos.get(node, (0.0, 0.0))
        node_x.append(x)
        node_y.append(y)
        freq = max(1, int(freq_map.get(node, 1)))
        strength = max(0.0, float(strength_map.get(node, 0.0)))
        size = 10 + 30 * (freq / max_freq)
        color_val = 20 + 80 * (strength / max_strength) if max_strength > 0 else 20
        node_sizes.append(size)
        node_colors.append(color_val)
        node_text.append(f"{node}<br>Frequency: {freq}<br>Co-occurrence Strength: {strength:.1f}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n for n in top_nodes],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Reds",
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title="Co-occurrence Strength",
                xanchor="left",
                titleside="right",
            ),
            line_width=1,
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Crime Type Co-Occurrence Network",
        title_x=0.5,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig


# ----------------------------
# Heatmap
# ----------------------------
def create_cooccurrence_heatmap(edges_df: pd.DataFrame) -> go.Figure:
    """
    Create a symmetric co-occurrence heatmap (Plotly) from edges_df.
    """
    if edges_df is None or edges_df.empty:
        return go.Figure()

    crimes = sorted(list(set(edges_df["source"].unique()) | set(edges_df["target"].unique())))
    if not crimes:
        return go.Figure()

    idx = {c: i for i, c in enumerate(crimes)}
    mat = np.zeros((len(crimes), len(crimes)), dtype=float)
    for _, row in edges_df.iterrows():
        s = row["source"]
        t = row["target"]
        w = float(row["weight"])
        i = idx.get(s)
        j = idx.get(t)
        if i is not None and j is not None:
            mat[i, j] = w
            mat[j, i] = w

    fig = px.imshow(
        mat,
        x=crimes,
        y=crimes,
        color_continuous_scale="Reds",
        labels=dict(color="Co-occurrence Count"),
        title="Crime Type Co-Occurrence Heatmap",
    )
    fig.update_layout(height=600, xaxis=dict(tickangle=-45))
    return fig
