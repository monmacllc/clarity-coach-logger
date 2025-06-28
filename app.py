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

st.set_page_config(page_title="Clarity Coach", layout="centered")

openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"
calendar_webhook_url = "https://hook.us2.make.com/nmd640nukq44ikms638z8w6yavqx1t3f"

logging.basicConfig(level=logging.INFO)

def extract_event_info(text):
    settings = {'PREFER_DAY_OF_MONTH': 'first', 'RELATIVE_BASE': datetime.now(pytz.utc)}
    matches = dateparser.search.search_dates(text, settings=settings)
    if not matches:
        now = datetime.now(pytz.utc)
        return now.isoformat(timespec='microseconds'), (now + timedelta(hours=1)).isoformat(timespec='microseconds'), None
    start = matches[0][1]
    end = start + timedelta(hours=1)
    return start.isoformat(timespec='microseconds'), end.isoformat(timespec='microseconds'), None

try:
    client = OpenAI(api_key=openai_api_key)
    client.models.list()
    openai_ok = True
except Exception as e:
    openai_ok = False
    st.error("OpenAI error")
    st.exception(e)

def load_sheet_data():
    sheet_ref = gs_client.open("Clarity Capture Log").sheet1
    values = sheet_ref.get_all_values()
    header = [h.strip() for h in values[0]]
    data = [dict(zip(header, row + [''] * (len(header) - len(row)))) for row in values[1:] if any(row)]
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True, infer_datetime_format=True)
    if df['Timestamp'].isna().any():
        st.warning("Some timestamps could not be parsed.")
    df = df.dropna(subset=['Timestamp'])
    df['Category'] = df['Category'].astype(str).str.lower().str.strip()
    df['Status'] = df.get('Status', 'Incomplete').astype(str).str.strip().str.capitalize()
    df['Priority'] = df.get('Priority', '').astype(str).str.strip()
    return sheet_ref, df

try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(os.getenv("GOOGLE_SERVICE_KEY")), scope)
    gs_client = gspread.authorize(creds)
    sheet, df = load_sheet_data()
    sheet_ok = True
except Exception as e:
    sheet_ok = False
    st.error("Google Sheet error")
    st.exception(e)

def render_category_form(category):
    with st.expander(category.upper()):
        with st.form(key=f"form_{category}"):
            input_text = st.text_area(f"Insight for {category}", height=100)
            submitted = st.form_submit_button(f"Log {category}")
            if submitted and input_text.strip():
                lines = [s.strip() for chunk in input_text.splitlines() for s in chunk.split(',') if s.strip()]
                for line in lines:
                    start, end, recurrence = extract_event_info(line)
                    entry = {
                        "timestamp": start,
                        "category": category.lower().strip(),
                        "insight": line,
                        "action_step": "",
                        "source": "Clarity Coach"
                    }
                    try:
                        requests.post(webhook_url, json=entry)
                    except Exception as e:
                        logging.warning(e)
                    cal_payload = {
                        "start": start,
                        "end": end,
                        "summary": line,
                        "category": category.lower().strip(),
                        "source": "Clarity Coach"
                    }
                    try:
                        requests.post(calendar_webhook_url, json=cal_payload)
                    except Exception as e:
                        logging.warning(e)
                st.success(f"Logged {len(lines)} insight(s)")
                time.sleep(2)
                global sheet, df
                sheet, df = load_sheet_data()
                st.write("Reloaded data:", df.tail(3))

if openai_ok and sheet_ok:
    tabs = st.tabs(["Log Clarity", "Recall Insights", "Clarity Chat"])
    categories = [
        "ccv","traditional real estate","stressors","co living","finances",
        "body mind spirit","wife","kids","family","quality of life","fun","giving back","misc"
    ]
    with tabs[0]:
        st.title("Log Clarity")
        for category in categories:
            render_category_form(category)
    with tabs[1]:
        st.title("Recall Insights")
        selected = st.multiselect("Categories", options=categories, default=categories)
        num_entries = st.slider("Entries", 5, 200, 50)
        sorted_df = df.sort_values(by="Timestamp", ascending=False).copy()
        filtered_df = sorted_df[sorted_df["Category"].isin([c.lower() for c in selected])]
        st.write(filtered_df[['Timestamp','Insight']].head(5))
        display_df = filtered_df.head(num_entries)
        for idx, row in display_df.iterrows():
            st.markdown(f"**{row['Category'].capitalize()}**: {row['Insight']}  \n*{row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}*")
    with tabs[2]:
        st.title("Clarity Chat")
        chat = st.text_area("Ask Clarity Coach:")
        if st.button("Ask"):
            if chat.strip():
                resp = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role":"system","content":"You are a clarity coach helping users improve their life."},
                        {"role":"user","content":chat}
                    ]
                )
                st.write(resp.choices[0].message.content)
