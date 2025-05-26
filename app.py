import streamlit as st
import requests
import json
from datetime import datetime, timedelta, date
from dateutil.parser import parse as dtparser
from openai import OpenAI
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
import pandas as pd

# --- SETUP ---
st.set_page_config(page_title="Clarity Coach", layout="centered")
openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"  # Google Sheets webhook
calendar_webhook_url = "https://hook.us2.make.com/nmd640nukq44ikms638z8w6yavqx1t3f"  # Google Calendar webhook

# --- CHECK OPENAI ACCESS ---
try:
    client = OpenAI(api_key=openai_api_key)
    client.models.list()
    openai_ok = True
except Exception as e:
    openai_ok = False
    st.error("❌ Failed to connect to OpenAI. Check your API key or billing status.")
    st.exception(e)

# --- GOOGLE SHEETS SETUP ---
try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    service_key_json = os.getenv("GOOGLE_SERVICE_KEY")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(service_key_json), scope)
    gs_client = gspread.authorize(creds)
    sheet = gs_client.open("Clarity Capture Log").sheet1

    test_values = sheet.get_all_values()
    st.success(f"✅ Successfully accessed Google Sheet. First row: {test_values[0]}")

    rows_raw = test_values
    header = rows_raw[0]
    data = [dict(zip(header, row + [''] * (len(header) - len(row)))) for row in rows_raw[1:] if any(row)]
    sheet_ok = True
except Exception as e:
    sheet_ok = False
    st.error("❌ Failed to connect to Google Sheet. Make sure the sheet is shared with your service account.")
    st.exception(e)

# --- TIME PARSER ---
def extract_event_time(insight, fallback_time=None):
    try:
        fallback = fallback_time or datetime.utcnow()
        dt = dtparser(insight, fuzzy=True, default=fallback)
        if dt < datetime.utcnow():
            return fallback.isoformat()
        return dt.isoformat()
    except:
        return (fallback_time or datetime.utcnow()).isoformat()

# --- STREAMLIT TABS ---
if openai_ok and sheet_ok:
    tabs = st.tabs(["🚀 Log Clarity", "🔍 Recall Insights", "💬 Clarity Chat"])

    # --- LOG TAB ---
    with tabs[0]:
        st.title("🧠 Clarity Coach")
        st.write("Enter your insights directly by category. Each form below logs to your sheet and calendar.")

        categories = [
            "ccv", "traditional real estate", "stressors", "co living", "finances",
            "body mind spirit", "wife", "kids", "family", "quality of life",
            "fun", "giving back", "misc"
        ]

        for category in categories:
            with st.expander(category.upper()):
                with st.form(key=f"form_{category}"):
                    input_text = st.text_area(f"Insight for {category}", key=f"input_{category}", height=100)
                    submitted = st.form_submit_button(f"Log {category} Insight")
                    if submitted and input_text.strip():
                        lines = [s.strip() for chunk in input_text.splitlines() for s in chunk.split(',') if s.strip()]
                        for line in lines:
                            parsed_time = extract_event_time(line)
                            timestamp = parsed_time if parsed_time else datetime.utcnow().isoformat()

                            entry = {
                                "timestamp": timestamp,
                                "category": category,
                                "insight": line,
                                "action_step": "",
                                "source": "Clarity Coach"
                            }

                            try:
                                requests.post(webhook_url, json=entry)
                            except Exception as e:
                                st.warning(f"Failed to log to Google Sheet: {e}")

                            try:
                                requests.post(calendar_webhook_url, json=entry)
                            except Exception as e:
                                st.warning(f"Failed to log to Google Calendar: {e}")

                        st.success(f"✅ Logged {len(lines)} insight(s) under {category}")
