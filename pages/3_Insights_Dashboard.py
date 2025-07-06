import streamlit as st
import json
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from datetime import datetime, timedelta
import altair as alt

# Page config
st.set_page_config(page_title="Clarity Coach - Insights Dashboard", layout="wide")

# Categories
categories_order = [
    "ccv",
    "traditional real estate",
    "n&ytg",
    "stressors",
    "co living",
    "finances",
    "body mind spirit",
    "wife",
    "kids",
    "family",
    "quality of life",
    "fun",
    "giving back",
    "misc",
]

# Load data
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    json.loads(os.getenv("GOOGLE_SERVICE_KEY")), scope
)
gs_client = gspread.authorize(creds)
sheet = gs_client.open("Clarity Capture Log").sheet1

values = sheet.get_all_values()
header = [h.strip() for h in values[0]]
data = []
for row in values[1:]:
    padded_row = row + [""] * (len(header) - len(row))
    record = dict(zip(header, padded_row))
    data.append(record)
df = pd.DataFrame(data)
df.columns = df.columns.str.strip()

df["CreatedAt"] = pd.to_datetime(df["CreatedAt"], errors="coerce", utc=True)
df["Status"] = df.get("Status", "Incomplete").astype(str).str.strip().str.capitalize()
df["Category"] = df["Category"].astype(str).str.lower().str.strip()

# Prepare timestamps
df_filtered = df.copy()
df_filtered["DaysAgo"] = df_filtered["CreatedAt"].apply(
    lambda d: (pd.Timestamp.utcnow() - d).days
)

# Assign timeframes
def bucket_label(days_ago):
    if days_ago <= 7:
        return "Last 7 Days"
    elif days_ago <= 14:
        return "Last 14 Days"
    elif days_ago <= 21:
        return "Last 21 Days"
    elif days_ago <= 30:
        return "Last 30 Days"
    else:
        return None

df_filtered["Timeframe"] = df_filtered["DaysAgo"].apply(bucket_label)
df_filtered = df_filtered[df_filtered["Timeframe"].notnull()]

# Aggregate counts
all_timeframes = ["Last 7 Days", "Last 14 Days", "Last 21 Days", "Last 30 Days"]
all_statuses = ["Complete", "Incomplete"]

entries_per_timeframe = (
    df_filtered.groupby(["Timeframe", "Status"])
    .size()
    .reset_index(name="Count")
)

idx = pd.MultiIndex.from_product(
    [all_timeframes, all_statuses],
    names=["Timeframe", "Status"]
)
entries_per_timeframe = (
    entries_per_timeframe
    .set_index(["Timeframe", "Status"])
    .reindex(idx, fill_value=0)
    .reset_index()
)

# UI
st.title("📊 Insights Dashboard")

st.markdown("### Entries by Timeframe")

col1, col2, col3 = st.columns(3)
show_14 = col1.checkbox("Include Last 14 Days")
show_21 = col2.checkbox("Include Last 21 Days")
show_30 = col3.checkbox("Include Last 30 Days")

selected_buckets = ["Last 7 Days"]
if show_14:
    selected_buckets.append("Last 14 Days")
if show_21:
    selected_buckets.append("Last 21 Days")
if show_30:
    selected_buckets.append("Last 30 Days")

entries_per_timeframe["Show"] = entries_per_timeframe["Timeframe"].isin(selected_buckets)
entries_per_timeframe["DisplayCount"] = entries_per_timeframe.apply(
    lambda row: row["Count"] if row["Show"] else 0,
    axis=1
)
entries_per_timeframe["label_text"] = entries_per_timeframe["DisplayCount"].apply(
    lambda x: str(x) if x > 0 else ""
)
entries_per_timeframe["Timeframe"] = pd.Categorical(
    entries_per_timeframe["Timeframe"],
    categories=all_timeframes,
    ordered=True
)

if entries_per_timeframe["DisplayCount"].sum() == 0:
    st.info("No entries to display.")
else:
    base = alt.Chart(entries_per_timeframe).encode(
        x=alt.X("Timeframe:N", sort=all_timeframes),
        y=alt.Y("DisplayCount:Q", title="Number of Entries"),
        color=alt.Color("Status:N"),
        tooltip=["Timeframe", "Status", "DisplayCount"]
    )
    bars = base.mark_bar()
    text_inside = base.transform_filter(
        alt.datum.DisplayCount >= 10
    ).mark_text(
        align="center",
        dy=5,
        color="black"
    ).encode(text="label_text:N")
    text_above = base.transform_filter(
        alt.datum.DisplayCount < 10
    ).mark_text(
        align="center",
        dy=-10,
        color="black"
    ).encode(text="label_text:N")
    chart = (bars + text_inside + text_above).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

# Completed entries by category
with st.expander("🥧 Completed Entries by Category (Last 30 Days)"):
    cutoff_date = pd.Timestamp.utcnow() - pd.Timedelta(days=30)
    completed_30 = df[
        (df["Status"] == "Complete") & (df["CreatedAt"] >= cutoff_date)
    ]
    counts = (
        completed_30.groupby("Category")
        .size()
        .reset_index(name="CompletedCount")
    )
    all_cats_df = pd.DataFrame({"Category": categories_order})
    merged_counts = pd.merge(
        all_cats_df,
        counts,
        on="Category",
        how="left"
    ).fillna(0)
    merged_counts["CompletedCount"] = merged_counts["CompletedCount"].astype(int)

    if merged_counts["CompletedCount"].sum() == 0:
        st.info("No completed entries in the past 30 days.")
    else:
        pie = alt.Chart(merged_counts).mark_arc(innerRadius=40).encode(
            theta=alt.Theta("CompletedCount:Q"),
            color=alt.Color("Category:N", sort=categories_order),
            tooltip=["Category", "CompletedCount"]
        ).properties(height=400)
        st.altair_chart(pie, use_container_width=True)

        bar = alt.Chart(merged_counts).mark_bar().encode(
            x=alt.X("CompletedCount:Q", title="Completed Entries"),
            y=alt.Y("Category:N", sort="-x"),
            tooltip=["Category", "CompletedCount"]
        ).properties(height=400, title="Completed Entries per Category")
        st.altair_chart(bar, use_container_width=True)

