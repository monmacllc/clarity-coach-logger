import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pytz
import dateparser
import dateparser.search
from openai import OpenAI
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import re
import logging
import time

# ✅ Streamlit Page Config
st.set_page_config(page_title="Clarity Coach", layout="centered")

# ✅ API Keys and Webhook URLs
openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"
calendar_webhook_url = "https://hook.us2.make.com/nmd640nukq44ikms638z8w6yavqx1t3f"

# ✅ Logging
logging.basicConfig(level=logging.INFO)

# ✅ Safe date parsing helper
def extract_event_info(text):
    settings = {"PREFER_DAY_OF_MONTH": "first", "RELATIVE_BASE": datetime.now(pytz.utc)}
    matches = dateparser.search.search_dates(text, settings=settings)
    now = datetime.now(pytz.utc)
    if not matches:
        return (
            now.isoformat(timespec="microseconds"),
            (now + timedelta(hours=1)).isoformat(timespec="microseconds"),
            None,
        )
    start = matches[0][1]
    if start.year > now.year + 2:
        start = now
    end = start + timedelta(hours=1)
    return (
        start.isoformat(timespec="microseconds"),
        end.isoformat(timespec="microseconds"),
        None,
    )

# ✅ OpenAI connectivity
try:
    client = OpenAI(api_key=openai_api_key)
    client.models.list()
    openai_ok = True
except Exception as e:
    openai_ok = False
    st.error("Failed to connect to OpenAI.")
    st.exception(e)

# ✅ Load Google Sheet data
def load_sheet_data():
    sheet_ref = gs_client.open("Clarity Capture Log").sheet1
    values = sheet_ref.get_all_values()
    header = [h.strip() for h in values[0]]

    # Ensure all expected columns exist
    required_columns = ["CreatedAt", "Status", "Priority", "Device"]
    for col in required_columns:
        if col not in header:
            header.append(col)
            sheet_ref.resize(rows=len(values), cols=len(header))

    data = []
    for row in values[1:]:
        padded_row = row + [""] * (len(header) - len(row))
        record = dict(zip(header, padded_row))
        data.append(record)

    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce", utc=True, infer_datetime_format=True
    )
    df["CreatedAt"] = pd.to_datetime(
        df["CreatedAt"], errors="coerce", utc=True, infer_datetime_format=True
    )
    # Fallback: if CreatedAt missing, use Timestamp
    df["CreatedAt"] = df["CreatedAt"].fillna(df["Timestamp"])
    df = df.dropna(subset=["CreatedAt"])
    df["Category"] = df["Category"].astype(str).str.lower().str.strip()
    df["Status"] = df.get("Status", "Incomplete").astype(str).str.strip().str.capitalize()
    df["Priority"] = df.get("Priority", "").astype(str).str.strip()
    df["Device"] = df.get("Device", "").astype(str).str.strip()
    return sheet_ref, df

# ✅ Connect to Google Sheets
try:
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        json.loads(os.getenv("GOOGLE_SERVICE_KEY")), scope
    )
    gs_client = gspread.authorize(creds)
    sheet, df = load_sheet_data()
    sheet_ok = True
except Exception as e:
    sheet_ok = False
    st.error("Failed to connect to Google Sheet.")
    st.exception(e)

# ✅ Form for logging insights
def render_category_form(category):
    with st.expander(category.upper()):
        with st.form(key=f"form_{category}"):
            input_text = st.text_area(
                f"Insight for {category}", key=f"input_{category}", height=100
            )
            submitted = st.form_submit_button(f"Log {category}")
            if submitted and input_text.strip():
                lines = [
                    s.strip()
                    for chunk in input
