# app/pages/6_Crime_Network.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from itertools import combinations

st.set_page_config(page_title="Crime Type Co-Occurrence Network", layout="wide")

st.title("Crime Type Co-Occurrence Network")
st.markdown("Explore how crime types tend to appear together across locations and time windows.")

# --- Import backend ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from components.cooccurrence_graph import (
        ensure_canonical_columns,
        get_available_options,
        filter_dataset_for_network,
        compute_cooccurrence_edges,
        compute_node_stats,
        create_network_figure,
        create_cooccurrence_heatmap,
    )
except ImportError as e:
    st.error(f"Failed to import components: {e}")
    st.stop()

# --- Load dataset ---
@st.cache_data(show_spinner=False)
def load_dataset(path):
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, low_memory=False)
    return df, path

DATASET_PATH = r"D:\crime_pattern_prediction\data\processed\dataset.csv"
df, path = load_dataset(DATASET_PATH)

if df is None:
    st.error(f"Dataset not found at: {DATASET_PATH}")
    st.stop()

st.success(f"Loaded {len(df):,} records from: {path}")

# --- Canonicalize columns ---
df = ensure_canonical_columns(df)

# --- Load sidebar options ---
options = get_available_options(df)

# ---------------------------------------------------------
# SIDEBAR (STATE / AREA / PART-OF-DAY REMOVED)
# ---------------------------------------------------------
st.sidebar.header("Network Filters")

# CITY (primary filter now)
city_options = options.get("cities", [])
selected_cities = st.sidebar.multiselect("City", city_options)

# YEAR
year_options = options.get("years", [])
selected_years = st.sidebar.multiselect("Year", year_options)

# MONTH
month_options = options.get("months", list(range(1, 13)))
selected_months = st.sidebar.multiselect("Month", month_options)

# TIME CATEGORY
time_cat_options = options.get("time_categories", [])
selected_time_cats = st.sidebar.multiselect("Time Category", time_cat_options)

st.sidebar.markdown("---")

# CO-OCCURRENCE SETTINGS
context = st.sidebar.selectbox(
    "Grouping Context",
    [
        "City + Time Category",
        "City + Part of Day",
        "City + Date",
        "Area Type + Time Category",
    ],
)

min_pair_support = st.sidebar.slider(
    "Minimum Joint Occurrences",
    min_value=1, max_value=20, value=1
)

max_nodes = st.sidebar.slider(
    "Max Crime Types in Network",
    min_value=5, max_value=40, value=20
)

show_heatmap = st.sidebar.checkbox("Show Co-Occurrence Heatmap", value=False)

st.sidebar.markdown("---")

# Apply button
apply_clicked = st.sidebar.button("Apply Filters")

# Reset
if st.sidebar.button("Reset Filters"):
    st.session_state.pop("last_apply", None)
    st.rerun()

# APPLY BUTTON HANDLING
if apply_clicked:
    st.session_state["last_apply"] = {
        "selected_cities": selected_cities,
        "selected_years": selected_years,
        "selected_months": selected_months,
        "selected_time_cats": selected_time_cats,
        "context": context,
        "min_pair_support": int(min_pair_support),
        "max_nodes": int(max_nodes),
        "show_heatmap": bool(show_heatmap),
    }

if "last_apply" not in st.session_state:
    st.info("Choose filters and click 'Apply Filters' to generate the network.")
    st.stop()

applied = st.session_state["last_apply"]

selected_cities = applied["selected_cities"]
selected_years = applied["selected_years"]
selected_months = applied["selected_months"]
selected_time_cats = applied["selected_time_cats"]

context = applied["context"]
min_pair_support = applied["min_pair_support"]
max_nodes = applied["max_nodes"]
show_heatmap = applied["show_heatmap"]

# ---------------------------------------------------------
# FILTER DATA (STATE/AREA/POD REMOVED)
# ---------------------------------------------------------
filtered_df = filter_dataset_for_network(
    df,
    states=None,
    cities=selected_cities if selected_cities else None,
    years=selected_years if selected_years else None,
    months=selected_months if selected_months else None,
    area_types=None,
    time_categories=selected_time_cats if selected_time_cats else None,
    part_of_day=None,
)

st.markdown(
    f"**Filtered Records:** {len(filtered_df):,} / {len(df):,} | "
    f"Distinct Crime Types: {filtered_df['Crime Type'].nunique()}"
)

if len(filtered_df) < 10:
    st.warning("Too few records after filtering. Loosen filters and reapply.")
    st.stop()

# ---------------------------------------------------------
# COMPUTE CO-OCCURRENCE
# ---------------------------------------------------------
with st.spinner("Computing crime-type co-occurrence network..."):
    edges_df = compute_cooccurrence_edges(
        filtered_df,
        grouping_context=context,
        min_pair_support=min_pair_support,
    )
    node_df = compute_node_stats(filtered_df, edges_df)

if edges_df.empty or node_df.empty:
    st.warning("No co-occurring crime pairs found with selected filters.")
    st.stop()

# ---------------------------------------------------------
# KPI
# ---------------------------------------------------------
st.header("Network Summary")

top_pair = edges_df.iloc[0]
top_label = f"{top_pair['source']} ↔ {top_pair['target']}"

c1, c2, c3 = st.columns(3)
c1.metric("Crime Types in Network", node_df["crime_type"].nunique())
c2.metric("Strong Associations (Edges)", len(edges_df))
c3.metric("Strongest Pair", top_label)

# ---------------------------------------------------------
# NETWORK GRAPH
# ---------------------------------------------------------
st.header("Co-Occurrence Network Graph")
fig_network = create_network_figure(edges_df, node_df, max_nodes=max_nodes)
st.plotly_chart(fig_network, use_container_width=True)

# ---------------------------------------------------------
# HEATMAP
# ---------------------------------------------------------
if show_heatmap:
    st.header("Co-Occurrence Heatmap")
    fig_heatmap = create_cooccurrence_heatmap(edges_df)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ---------------------------------------------------------
# DETAIL TABLES
# ---------------------------------------------------------
st.header("Detailed Crime Associations")
tab1, tab2 = st.tabs(["Top Pairs", "Central Crime Types"])

with tab1:
    st.dataframe(
        edges_df.rename(columns={
            "source": "Crime A",
            "target": "Crime B",
            "weight": "Joint Occurrences"
        }),
        use_container_width=True
    )

with tab2:
    st.dataframe(
        node_df.rename(columns={
            "crime_type": "Crime Type",
            "frequency": "Frequency",
            "degree_strength": "Co-Occurrence Strength"
        }),
        use_container_width=True
    )
