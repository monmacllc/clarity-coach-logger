import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pytz
from dateutil.parser import parse as dtparser
import dateparser
import dateparser.search
from openai import OpenAI
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
import pandas as pd
import re
import logging

st.set_page_config(page_title="Clarity Coach", layout="centered")

openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"
calendar_webhook_url = "https://hook.us2.make.com/nmd640nukq44ikms638z8w6yavqx1t3f"

logging.basicConfig(level=logging.INFO)

# Helper: Natural language datetime extraction
def extract_event_info(text):
    settings = {'PREFER_DAY_OF_MONTH': 'first', 'RELATIVE_BASE': datetime.now(pytz.utc)}
    matches = dateparser.search.search_dates(text, settings=settings)
    if matches:
        start = matches[0][1]
        time_range = re.search(r'(\d{1,2})(?::\d{2})?\s*[-to–]\s*(\d{1,2})(?::\d{2})?', text)
        if time_range:
            try:
                start_hour = int(time_range.group(1))
                end_hour = int(time_range.group(2))
                end = start.replace(hour=end_hour)
            except:
                end = start + timedelta(hours=1)
        else:
            end = start + timedelta(hours=1)
        return start.isoformat(), end.isoformat(), None
    now = datetime.now(pytz.utc)
    return now.isoformat(), (now + timedelta(hours=1)).isoformat(), None

try:
    client = OpenAI(api_key=openai_api_key)
    client.models.list()
    openai_ok = True
except Exception as e:
    openai_ok = False
    st.error("Failed to connect to OpenAI. Check your API key or billing status.")
    st.exception(e)

try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    service_key_json = os.getenv("GOOGLE_SERVICE_KEY")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(service_key_json), scope)
    gs_client = gspread.authorize(creds)
    sheet = gs_client.open("Clarity Capture Log").sheet1
    test_values = sheet.get_all_values()
    header = [h.strip() for h in test_values[0]]

    if 'Priority' not in header:
        header.append('Priority')
        sheet.resize(rows=len(test_values), cols=len(header))
        updates = [[''] for _ in range(2, len(test_values) + 1)]
        sheet.update(f'{chr(65 + len(header) - 1)}2:{chr(65 + len(header) - 1)}{len(test_values)}', updates)

    data = [dict(zip(header, row + [''] * (len(header) - len(row)))) for row in test_values[1:] if any(row)]
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['Timestamp'])
    df['Category'] = df['Category'].astype(str).str.lower().str.strip()
    df['Status'] = df.get('Status', 'Incomplete').astype(str).str.strip().str.capitalize()
    df['Priority'] = df.get('Priority', '').astype(str).str.strip()
    sheet_ok = True
except Exception as e:
    sheet_ok = False
    st.error("Failed to connect to Google Sheet.")
    st.exception(e)

def render_category_form(category):
    with st.expander(category.upper()):
        with st.form(key=f"form_{category}"):
            input_text = st.text_area(f"Insight for {category}", key=f"input_{category}", height=100)
            submitted = st.form_submit_button(f"Log {category} Insight")
            if submitted and input_text.strip():
                lines = [s.strip() for chunk in input_text.splitlines() for s in chunk.split(',') if s.strip()]
                for line in lines:
                    start, end, recurrence = extract_event_info(line)
                    entry = {"timestamp": start, "category": category, "insight": line, "action_step": "", "source": "Clarity Coach"}
                    logging.info(f"Logging entry: {entry}")
                    try:
                        requests.post(webhook_url, json=entry)
                    except: pass
                    cal_payload = {"start": start, "end": end, "summary": line, "category": category, "source": "Clarity Coach"}
                    if recurrence:
                        cal_payload["recurrence"] = recurrence
                    try:
                        requests.post(calendar_webhook_url, json=cal_payload)
                    except: pass
                st.success(f"Logged {len(lines)} insight(s) under {category}")

                # Refresh Google Sheet data
                test_values = sheet.get_all_values()
                data = [dict(zip(header, row + [''] * (len(header) - len(row)))) for row in test_values[1:] if any(row)]
                df = pd.DataFrame(data)
                df.columns = df.columns.str.strip()
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
                df = df.dropna(subset=['Timestamp'])
                df['Category'] = df['Category'].astype(str).str.lower().str.strip()
                df['Status'] = df.get('Status', 'Incomplete').astype(str).str.strip().str.capitalize()
                df['Priority'] = df.get('Priority', '').astype(str).str.strip()

if openai_ok and sheet_ok:
    tabs = st.tabs(["Log Clarity", "Recall Insights", "Clarity Chat"])

    with tabs[0]:
        st.title("Clarity Coach")
        categories = ["ccv", "traditional real estate", "stressors", "co living", "finances", "body mind spirit", "wife", "kids", "family", "quality of life", "fun", "giving back", "misc"]
        for category in categories:
            render_category_form(category)

    with tabs[1]:
        st.title("Recall Insights")
        standard_categories = ["ccv", "traditional real estate", "stressors", "co living", "finances", "body mind spirit", "wife", "kids", "family", "quality of life", "fun", "giving back", "misc"]
        select_all = st.checkbox("Select All Categories", value=True)
        selected_categories = st.multiselect("Select Categories", options=standard_categories, default=standard_categories if select_all else [])
        days = st.slider("Days to look back", 1, 90, 30)
        recall_df = df[df['Timestamp'] > datetime.now(pytz.utc) - timedelta(days=days)]
        recall_df = recall_df[recall_df['Category'].isin(selected_categories)]
        recall_df = recall_df.sort_values(by='Timestamp', ascending=False)
        show_completed = st.sidebar.checkbox("Show Completed Items", False)
        debug_mode = st.sidebar.checkbox("Debug Mode", False)

        if debug_mode:
            st.subheader("Raw Data")
            st.dataframe(df)

        if not show_completed:
            recall_df = recall_df[recall_df['Status'] != 'Complete']

        grouped = recall_df.groupby('Category')
        for category, group in grouped:
            st.subheader(category.upper())
            group = group.sort_values(by='Timestamp', ascending=False)
            for idx, row in group.iterrows():
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    marked = st.checkbox(f"{row['Insight']} ({row['Timestamp'].strftime('%m/%d/%Y')})", key=f"check_{idx}")
                with col2:
                    is_starred = str(row.get('Priority', '')).strip().lower() == 'yes'
                    starred = st.checkbox("⭐", value=is_starred, key=f"star_{idx}")
                if marked and row['Status'] != 'Complete':
                    row_index = df[df['Insight'] == row['Insight']].index[0] + 2
                    sheet.update_cell(row_index, df.columns.get_loc("Status") + 1, "Complete")
                    st.success("Marked as complete")
                    st.success("Marked as complete")
                if starred != is_starred:
                    val = "Yes" if starred else ""
                    row_index = df[df['Insight'] == row['Insight']].index[0] + 2
                    col_num = df.columns.get_loc("Priority") + 1
                    try:
                        sheet.update_cell(row_index, col_num, val)
                        st.info(f"Updated Priority at row {row_num}, column {col_num} to '{val}'")
                    except Exception as e:
                        st.warning(f"Failed to update Priority at row {row_num}, column {col_num}: {e}")
