import streamlit as st
import requests
import json
from datetime import datetime, timedelta, date, time as dtime
from dateutil.parser import parse as dtparser
from openai import OpenAI
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
import pandas as pd
import re
import calendar

# --- SETUP ---
st.set_page_config(page_title="Clarity Coach", layout="centered")
openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"
calendar_webhook_url = "https://hook.us2.make.com/nmd640nukq44ikms638z8w6yavqx1t3f"

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
    header = [h.strip() for h in rows_raw[0]]
    data = [dict(zip(header, row + [''] * (len(header) - len(row)))) for row in rows_raw[1:] if any(row)]
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    if 'Status' not in df.columns:
        df['Status'] = 'Incomplete'
    else:
        df['Status'] = df['Status'].astype(str).str.strip().str.capitalize()
    df['Category'] = df['Category'].astype(str).str.lower().str.strip()
    sheet_ok = True
except Exception as e:
    sheet_ok = False
    st.error("❌ Failed to connect to Google Sheet. Make sure the sheet is shared with your service account.")
    st.exception(e)

# --- TIME + RECURRENCE PARSER ---
def extract_event_info(insight, fallback_time=None):
    try:
        base = fallback_time or datetime.utcnow()
        text = insight.lower()

        if re.search(r"\btmr\b|\btomorrow\b", text):
            base += timedelta(days=1)

        weekdays = list(calendar.day_name)
        for i, day in enumerate(weekdays):
            if re.search(rf"\bnext {day.lower()}\b", text):
                delta = (i - base.weekday() + 7) % 7 + 7
                base += timedelta(days=delta)
            elif re.search(rf"\b{day.lower()}\b", text):
                delta = (i - base.weekday() + 7) % 7
                if delta == 0:
                    delta = 7
                base += timedelta(days=delta)

        match = re.search(r'(\d{1,2})([:\.]?(\d{2}))?\s*(am|pm)?\s*(-|to)\s*(\d{1,2})([:\.]?(\d{2}))?\s*(am|pm)?', insight)
        if match:
            sh = int(match.group(1))
            sm = int(match.group(3)) if match.group(3) else 0
            eh = int(match.group(6))
            em = int(match.group(8)) if match.group(8) else 0

            def infer_meridiem(hour, ampm):
                if ampm == "am": return hour if hour != 12 else 0
                if ampm == "pm": return hour + 12 if hour < 12 else hour
                return hour + 12 if hour < 7 else hour

            start_hour = infer_meridiem(sh, match.group(4))
            end_hour = infer_meridiem(eh, match.group(9))
            start_dt = datetime.combine(base.date(), dtime(start_hour, sm))
            end_dt = datetime.combine(base.date(), dtime(end_hour, em))
        else:
            match = re.search(r'(\d{1,2})([:\.]?(\d{2}))?\s*(am|pm)?', insight)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(3)) if match.group(3) else 0
                ampm = match.group(4)
                def infer(hour, ampm):
                    if ampm == "am": return hour if hour != 12 else 0
                    if ampm == "pm": return hour + 12 if hour < 12 else hour
                    return hour + 12 if hour < 7 else hour
                start_hour = infer(hour, ampm)
                start_dt = datetime.combine(base.date(), dtime(start_hour, minute))
                end_dt = start_dt + timedelta(hours=1)
            else:
                start_dt = datetime.combine(base.date(), dtime(3, 0))
                end_dt = start_dt + timedelta(hours=1)

        if start_dt < datetime.utcnow():
            start_dt += timedelta(days=1)
            end_dt += timedelta(days=1)

        recurrence = None
        if "every day" in text or "each day" in text:
            recurrence = ["RRULE:FREQ=DAILY"]
        elif "every weekday" in text:
            recurrence = ["RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"]
        elif re.search(r"every (mon|tue|wed|thu|fri|sat|sun)", text):
            day_map = {"mon":"MO","tue":"TU","wed":"WE","thu":"TH","fri":"FR","sat":"SA","sun":"SU"}
            day_key = re.search(r"every (mon|tue|wed|thu|fri|sat|sun)", text).group(1)
            recurrence = [f"RRULE:FREQ=WEEKLY;BYDAY={day_map[day_key]}"]

        return start_dt.isoformat(), end_dt.isoformat(), recurrence

    except Exception as e:
        fallback = fallback_time or datetime.utcnow()
        return fallback.isoformat(), (fallback + timedelta(hours=1)).isoformat(), None
