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
import logging
import time

# Page config
st.set_page_config(page_title="Clarity Coach", layout="centered")

# Timezone for display
local_tz = pytz.timezone("US/Pacific")

# API Keys and Webhooks
openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"
calendar_webhook_url = "https://hook.us2.make.com/nmd640nukq44ikms638z8w6yavqx1t3f"

# Logging
logging.basicConfig(level=logging.INFO)

# Date parsing function
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

    if start.year < 1900 or start.year > 2100:
        start = now

    end = start + timedelta(hours=1)
    return (
        start.isoformat(timespec="microseconds"),
        end.isoformat(timespec="microseconds"),
        None,
    )

# OpenAI connection
try:
    client = OpenAI(api_key=openai_api_key)
    client.models.list()
    openai_ok = True
except Exception as e:
    openai_ok = False
    st.error("OpenAI error")
    st.exception(e)

# Load Google Sheets data
def load_sheet_data():
    sheet_ref = gs_client.open("Clarity Capture Log").sheet1
    values = sheet_ref.get_all_values()
    header = [h.strip() for h in values[0]]

    required_columns = ["CreatedAt", "Status", "Priority", "Device", "RowIndex"]
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

    def parse_timestamp(value):
        try:
            if pd.isnull(value):
                return pd.NaT
            if isinstance(value, (float, int)):
                return pd.to_datetime("1899-12-30") + pd.to_timedelta(value, unit="D")
            return pd.to_datetime(value, utc=True, errors="coerce")
        except:
            return pd.NaT

    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="ignore")
    df["Timestamp"] = df["Timestamp"].apply(parse_timestamp)

    df["CreatedAt"] = pd.to_datetime(df["CreatedAt"], errors="coerce", utc=True)
    df["CreatedAt"] = df["CreatedAt"].fillna(df["Timestamp"])

    df = df.dropna(subset=["CreatedAt"])

    df["RowIndex"] = pd.to_numeric(df["RowIndex"], errors="coerce")

    df["Category"] = df["Category"].astype(str).str.lower().str.strip()
    df["Status"] = df.get("Status", "Incomplete").astype(str).str.strip().str.capitalize()
    df["Priority"] = df.get("Priority", "").astype(str).str.strip()
    df["Device"] = df.get("Device", "").astype(str).str.strip()
    return sheet_ref, df

# Connect to Google Sheets
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
    st.error("Google Sheets error")
    st.exception(e)

# Log form per category
def render_category_form(category, clarity_debug, show_starred):
    with st.expander(category.upper()):
        with st.form(key=f"form_{category}"):
            input_text = st.text_area(f"Insight for {category}", height=100)
            submitted = st.form_submit_button(f"Log {category}")
            if submitted and input_text.strip():
                lines = [
                    s.strip()
                    for chunk in input_text.splitlines()
                    for s in chunk.split(",")
                    if s.strip()
                ]
                for line in lines:
                    start, end, recurrence = extract_event_info(line)
                    created_at = datetime.utcnow().isoformat(timespec="microseconds")
                    entry = {
                        "timestamp": start,
                        "created_at": created_at,
                        "category": category.lower().strip(),
                        "insight": line,
                        "action_step": "",
                        "source": "Clarity Coach",
                        "status": "Incomplete",
                        "priority": "",
                        "device": "Web",
                    }

                    if clarity_debug:
                        st.write("🚨 Payload sent to webhook:")
                        st.json(entry)

                    try:
                        requests.post(webhook_url, json=entry)
                    except Exception as e:
                        logging.warning(e)
                    cal_payload = {
                        "start": start,
                        "end": end,
                        "summary": line,
                        "category": category.lower().strip(),
                        "source": "Clarity Coach",
                    }
                    try:
                        requests.post(calendar_webhook_url, json=cal_payload)
                    except Exception as e:
                        logging.warning(e)

                st.success(f"Logged {len(lines)} insight(s)")
                time.sleep(3)
                global sheet, df
                sheet, df = load_sheet_data()
                if clarity_debug:
                    if show_starred:
                        df = df[df["Priority"].str.lower() == "yes"]
                    st.write("Latest entries:", df.tail(5))

# Main tabs
if openai_ok and sheet_ok:
    tabs = st.tabs(["Clarity Log", "Recall Insights", "Clarity Chat"])

    # Clarity Log Tab
    with tabs[0]:
        st.title("Clarity Coach")
        clarity_debug = st.sidebar.checkbox("Clarity Log Debug Mode", False)
        show_starred = st.sidebar.checkbox("Show Starred Entries Only", False)
        categories = [
            "ccv",
            "traditional real estate",
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
        for category in categories:
            render_category_form(category, clarity_debug, show_starred)

    # Recall Insights Tab
    with tabs[1]:
        st.title("Recall Insights")
        selected = st.multiselect("Categories", options=categories, default=categories)
        num_entries = st.slider("Entries to display", 5, 200, 50)
        show_completed = st.sidebar.checkbox("Show Completed", False)
        sho
