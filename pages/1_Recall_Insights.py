import streamlit as st
import json
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Page config
st.set_page_config(page_title="Clarity Coach - Recall Insights", layout="wide")

# Timezone
local_tz = pytz.timezone("US/Pacific")

# Categories
categories = [
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

# Connect to Google Sheets
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    json.loads(os.getenv("GOOGLE_SERVICE_KEY")), scope
)
gs_client = gspread.authorize(creds)
sheet = gs_client.open("Clarity Capture Log").sheet1

# Load data
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
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
df["RowIndex"] = pd.to_numeric(df["RowIndex"], errors="coerce")
df["Category"] = df["Category"].astype(str).str.lower().str.strip()
df["Status"] = df.get("Status", "Incomplete").astype(str).str.strip().str.capitalize()
df["Priority"] = df.get("Priority", "").astype(str).str.strip()

# Remove completed entries older than 14 days
cutoff_14 = pd.Timestamp.utcnow() - pd.Timedelta(days=14)
df = df[
    ~(
        (df["Status"] == "Complete") &
        (df["CreatedAt"] < cutoff_14)
    )
]

# Sidebar filters
st.sidebar.header("Filters")
show_completed = st.sidebar.checkbox("Show Completed", False)
show_starred = st.sidebar.checkbox("Show Starred Only", False)
show_timestamps = st.sidebar.checkbox("Show Timestamps", False)
debug_mode = st.sidebar.checkbox("Debug Mode", False)

# Filter UI
st.title("🔍 Recall Insights")
selected = st.multiselect("Categories", options=categories, default=categories)
num_entries = st.slider("Entries to display", 5, 200, 50)

# Filter DataFrame
sorted_df = df.sort_values(by="RowIndex", ascending=False).copy()
filtered_df = sorted_df[
    sorted_df["Category"].isin([c.lower().strip() for c in selected])
]

if not show_completed:
    filtered_df = filtered_df[filtered_df["Status"] != "Complete"]
if show_starred:
    filtered_df = filtered_df[filtered_df["Priority"].str.lower() == "yes"]

display_df = filtered_df.head(num_entries)

if debug_mode:
    st.subheader("🚨 Debug Data")
    st.dataframe(display_df)

# Show entries grouped by category
for category in categories:
    cat_df = display_df[
        display_df["Category"] == category.lower().strip()
    ]
    if cat_df.empty:
        continue

    st.subheader(category.capitalize())
    for idx, row in cat_df.iterrows():
        created_at_str = (
            row["CreatedAt"].astimezone(local_tz).strftime("%Y-%m-%d %I:%M %p %Z")
            if pd.notnull(row["CreatedAt"])
            else "No Log Time"
        )
        label_text = row["Insight"]

        col1, col2 = st.columns([0.85, 0.15])

        with col1:
            marked = st.checkbox(
                label_text,
                key=f"check_{idx}",
                value=row["Status"] == "Complete",
            )
            if show_timestamps:
                st.markdown(f"**Logged:** {created_at_str}")

        with col2:
            starred = st.checkbox(
                "⭐",
                value=row["Priority"].lower() == "yes",
                key=f"star_{idx}"
            )

        if marked and row["Status"] != "Complete":
            row_index = df[df["Insight"] == row["Insight"]].index[0] + 2
            sheet.update_cell(row_index, df.columns.get_loc("Status") + 1, "Complete")
            st.success("Marked as complete")

        if starred and row["Priority"].lower() != "yes":
            row_index = df[df["Insight"] == row["Insight"]].index[0] + 2
            sheet.update_cell(row_index, df.columns.get_loc("Priority") + 1, "Yes")
            st.info("Starred")
        elif not starred and row["Priority"].lower() == "yes":
            row_index = df[df["Insight"] == row["Insight"]].index[0] + 2
            sheet.update_cell(row_index, df.columns.get_loc("Priority") + 1, "")
            st.info("Unstarred")

