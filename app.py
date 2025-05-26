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

        match = re.search(r'(\d{1,2})([:\.]?\d{0,2})?\s*(-|to)?\s*(\d{1,2})?(?:[:\.]?(\d{0,2}))?', insight)
        if match:
            start_hr = int(match.group(1))
            start_min = int(match.group(2)[1:]) if match.group(2) else 0

            def infer_meridiem(hour):
                if hour >= 7 and hour <= 11:
                    return hour
                elif hour >= 1 and hour <= 6:
                    return hour + 12
                else:
                    return hour

            start_hour = infer_meridiem(start_hr)
            start_dt = datetime.combine(base.date(), dtime(start_hour, start_min))
            if start_dt < datetime.utcnow():
                start_dt += timedelta(days=1)
        else:
            start_dt = dtparser(insight, fuzzy=True, default=base)
            if start_dt < datetime.utcnow():
                start_dt += timedelta(days=1)

        recurrence = None
        if "every day" in text or "each day" in text:
            recurrence = ["RRULE:FREQ=DAILY"]
        elif "every weekday" in text:
            recurrence = ["RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"]
        elif re.search(r"every (mon|tue|wed|thu|fri|sat|sun)", text):
            day_map = {"mon":"MO","tue":"TU","wed":"WE","thu":"TH","fri":"FR","sat":"SA","sun":"SU"}
            day_key = re.search(r"every (mon|tue|wed|thu|fri|sat|sun)", text).group(1)
            recurrence = [f"RRULE:FREQ=WEEKLY;BYDAY={day_map[day_key]}"]

        return start_dt.isoformat(), recurrence

    except Exception as e:
        return (fallback_time or datetime.utcnow()).isoformat(), None

# --- STREAMLIT TABS ---
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
                            timestamp, recurrence = extract_event_info(line)
                            entry = {"timestamp": timestamp, "category": category, "insight": line, "action_step": "", "source": "Clarity Coach"}

                            try:
                                requests.post(webhook_url, json=entry)
                            except Exception as e:
                                st.warning(f"Failed to log to Google Sheet: {e}")

                            cal_payload = entry.copy()
                            if recurrence:
                                cal_payload["recurrence"] = recurrence
                            try:
                                requests.post(calendar_webhook_url, json=cal_payload)
                            except Exception as e:
                                st.warning(f"Failed to log to Google Calendar: {e}")

                        st.success(f"✅ Logged {len(lines)} insight(s) under {category}")

    with tabs[1]:
        st.title("🔍 Recall Insights")
        selected_categories = st.multiselect("Select Categories", sorted(df['Category'].unique()), default=sorted(df['Category'].unique()))
        days = st.slider("Days to look back", 1, 90, 30)
        recall_df = df[df['Timestamp'] > datetime.utcnow() - timedelta(days=days)]
        recall_df = recall_df[recall_df['Category'].isin(selected_categories)]
        show_completed = st.sidebar.checkbox("Show Completed Items", value=True)
        debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

        if debug_mode:
            st.subheader("📋 All Rows (Raw Data)")
            st.dataframe(df)

        if not show_completed:
            recall_df = recall_df[recall_df['Status'] != 'Complete']

        grouped = recall_df.groupby('Category')
        for category, group in grouped:
            st.subheader(category.upper())
            for i, row in group.iterrows():
                insight = row['Insight']
                ts = row['Timestamp']
                checkbox = st.checkbox(f"{insight} ({ts.date()})", key=f"check_{i}")
                if checkbox and row['Status'] != 'Complete':
                    sheet.update_cell(i + 2, df.columns.get_loc("Status") + 1, "Complete")
                    st.success("Marked as complete")

        if st.button("🧠 Summarize Insights"):
            if not recall_df.empty:
                insight_texts = [f"- {row['Insight']} ({row['Category']})" for _, row in recall_df.iterrows() if pd.notnull(row['Insight']) and pd.notnull(row['Category'])]
                prompt = "Summarize these clarity insights by category:\n\n" + "\n".join(insight_texts)
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "system", "content": "You are Clarity Coach. Return a structured, insightful summary by category."}, {"role": "user", "content": prompt}]
                )
                st.markdown("### 🧠 Clarity Summary")
                st.write(response.choices[0].message.content)
            else:
                st.info("No insights available to summarize. Try adjusting filters.")

        st.markdown("---")
        st.header("📈 Completion Metrics")
        df['Week'] = df['Timestamp'].dt.to_period("W").apply(lambda r: r.start_time.date())
        completion_trend = df[df['Status'] == 'Complete'].groupby('Week').size().reset_index(name='Completed')
        fig1 = px.bar(completion_trend, x='Week', y='Completed', title='Weekly Completed Insights')
        st.plotly_chart(fig1, use_container_width=True)
        category_summary = df[df['Status'] == 'Complete'].groupby('Category').size().reset_index(name='Completed')
        fig2 = px.pie(category_summary, names='Category', values='Completed', title='Completed Insights by Category')
        st.plotly_chart(fig2, use_container_width=True)
        total = len(df)
        completed = len(df[df['Status'] == 'Complete'])
        if total > 0:
            st.metric("Completion Rate", f"{(completed / total * 100):.1f}%")
        else:
            st.metric("Completion Rate", "0.0%")

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
            reply = response.choices[0].message.content
            st.chat_message("assistant").write(reply)
        else:
            st.info("You haven't shared any recent brain dumps or insights yet for me to analyze and identify the top 80/20 priorities. Please provide your thoughts, tasks, or notes for today, and I can help determine the key focus areas.")
