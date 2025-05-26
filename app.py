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

st.set_page_config(page_title="Clarity Coach", layout="centered")
openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"
calendar_webhook_url = "https://hook.us2.make.com/nmd640nukq44ikms638z8w6yavqx1t3f"

try:
    client = OpenAI(api_key=openai_api_key)
    client.models.list()
    openai_ok = True
except Exception as e:
    openai_ok = False
    st.error("❌ Failed to connect to OpenAI. Check your API key or billing status.")
    st.exception(e)

try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    service_key_json = os.getenv("GOOGLE_SERVICE_KEY")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(service_key_json), scope)
    gs_client = gspread.authorize(creds)
    sheet = gs_client.open("Clarity Capture Log").sheet1
    test_values = sheet.get_all_values()
    header = [h.strip() for h in test_values[0]]
    data = [dict(zip(header, row + [''] * (len(header) - len(row)))) for row in test_values[1:] if any(row)]
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
    st.error("❌ Failed to connect to Google Sheet.")
    st.exception(e)

def extract_event_info(insight, fallback_time=None):
    try:
        base = fallback_time or datetime.utcnow()
        text = insight.lower()

        if re.search(r"\btmr\b|\btomorrow\b", text):
            base += timedelta(days=1)

        weekdays = list(calendar.day_name)
        for i, day in enumerate(weekdays):
            if re.search(rf"\bnext {day.lower()}\b", text):
                base += timedelta(days=(i - base.weekday() + 7) % 7 + 7)
            elif re.search(rf"\b{day.lower()}\b", text):
                base += timedelta(days=(i - base.weekday() + 7) % 7 or 7)

        match = re.search(r'(\d{1,2})([:\.]?(\d{2}))?\s*(am|pm)?\s*(-|to)\s*(\d{1,2})([:\.]?(\d{2}))?\s*(am|pm)?', text)
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
            match = re.search(r'(\d{1,2})([:\.]?(\d{2}))?\s*(am|pm)?', text)
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
    except:
        fallback = fallback_time or datetime.utcnow()
        return fallback.isoformat(), (fallback + timedelta(hours=1)).isoformat(), None

if openai_ok and sheet_ok:
    tabs = st.tabs(["🚀 Log Clarity", "🔍 Recall Insights", "💬 Clarity Chat"])

    with tabs[0]:
        st.title("🧠 Clarity Coach")
        categories = ["ccv", "traditional real estate", "stressors", "co living", "finances", "body mind spirit", "wife", "kids", "family", "quality of life", "fun", "giving back", "misc"]
        for category in categories:
            with st.expander(category.upper()):
                with st.form(key=f"form_{category}"):
                    input_text = st.text_area(f"Insight for {category}", key=f"input_{category}", height=100)
                    submitted = st.form_submit_button(f"Log {category} Insight")
                    if submitted and input_text.strip():
                        lines = [s.strip() for chunk in input_text.splitlines() for s in chunk.split(',') if s.strip()]
                        for line in lines:
                            start, end, recurrence = extract_event_info(line)
                            entry = {"timestamp": start, "category": category, "insight": line, "action_step": "", "source": "Clarity Coach"}
                            try: requests.post(webhook_url, json=entry)
                            except: pass
                            cal_payload = {"start": start, "end": end, "category": category, "insight": line, "action_step": "", "source": "Clarity Coach"}
                            if recurrence: cal_payload["recurrence"] = recurrence
                            try: requests.post(calendar_webhook_url, json=cal_payload)
                            except: pass
                        st.success(f"✅ Logged {len(lines)} insight(s) under {category}")

    with tabs[1]:
        st.title("🔍 Recall Insights")
        selected_categories = st.multiselect("Select Categories", sorted(df['Category'].unique()), default=sorted(df['Category'].unique()))
        days = st.slider("Days to look back", 1, 90, 30)
        recall_df = df[df['Timestamp'] > datetime.utcnow() - timedelta(days=days)]
        recall_df = recall_df[recall_df['Category'].isin(selected_categories)]
        show_completed = st.sidebar.checkbox("Show Completed Items", True)
        debug_mode = st.sidebar.checkbox("Debug Mode", False)
        if debug_mode:
            st.subheader("📋 Raw Data")
            st.dataframe(df)
        if not show_completed:
            recall_df = recall_df[recall_df['Status'] != 'Complete']
        grouped = recall_df.groupby('Category')
        for category, group in grouped:
            st.subheader(category.upper())
            for i, row in group.iterrows():
                if st.checkbox(f"{row['Insight']} ({row['Timestamp'].date()})", key=f"check_{i}") and row['Status'] != 'Complete':
                    sheet.update_cell(i + 2, df.columns.get_loc("Status") + 1, "Complete")
                    st.success("Marked as complete")
        if st.button("🧠 Summarize Insights"):
            if not recall_df.empty:
                insights = [f"- {row['Insight']} ({row['Category']})" for _, row in recall_df.iterrows() if pd.notnull(row['Insight']) and pd.notnull(row['Category'])]
                prompt = "Summarize these clarity insights by category:\n\n" + "\n".join(insights)
                response = client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role": "system", "content": "You are Clarity Coach."}, {"role": "user", "content": prompt}])
                st.markdown("### 🧠 Clarity Summary")
                st.write(response.choices[0].message.content)

    with tabs[2]:
        st.title("💬 Clarity Chat")
        recent_df = df[df['Timestamp'] > datetime.utcnow() - timedelta(days=30)]
        recent_insights = [f"- {row['Insight']} ({row['Category']})" for _, row in recent_df.iterrows() if pd.notnull(row['Insight'])]
        chat_input = st.chat_input("Type your clarity dump, summary request, or question...")
        if chat_input or recent_insights:
            st.chat_message("user").write(chat_input or "Analyze my recent clarity insights")
            system_prompt = "You are Clarity Coach. Help the user gain focus by analyzing the following insights. Identify themes, patterns, and top 80/20 priorities. Provide a short, clear strategic summary."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": chat_input or "\n".join(recent_insights)}]
            response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
            st.chat_message("assistant").write(response.choices[0].message.content)
        else:
            st.info("You haven't shared any recent brain dumps or insights yet. Add your thoughts to get focused feedback.")
