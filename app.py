import streamlit as st
import requests
import json
from datetime import datetime, timedelta
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

st.set_page_config(page_title="Clarity Coach", layout="centered")
openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"
calendar_webhook_url = "https://hook.us2.make.com/nmd640nukq44ikms638z8w6yavqx1t3f"

# Helper: Natural language datetime extraction
def extract_event_info(text):
    matches = dateparser.search.search_dates(text, settings={'PREFER_DATES_FROM': 'future'})
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
    return None, None, None

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
    st.error("Failed to connect to Google Sheet.")
    st.exception(e)

if openai_ok and sheet_ok:
    tabs = st.tabs(["Log Clarity", "Recall Insights", "Clarity Chat"])

    with tabs[0]:
        st.title("Clarity Coach")
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
                            cal_payload = {"start": start, "end": end, "summary": line, "category": category, "source": "Clarity Coach"}
                            if recurrence: cal_payload["recurrence"] = recurrence
                            try: requests.post(calendar_webhook_url, json=cal_payload)
                            except: pass
                        st.success(f"Logged {len(lines)} insight(s) under {category}")

    with tabs[1]:
        st.title("Recall Insights")

        standard_categories = ["ccv", "traditional real estate", "stressors", "co living", "finances", "body mind spirit", "wife", "kids", "family", "quality of life", "fun", "giving back", "misc"]
        select_all = st.checkbox("Select All Categories", value=True)
        selected_categories = st.multiselect(
            "Select Categories",
            options=standard_categories,
            default=standard_categories if select_all else []
        )

        days = st.slider("Days to look back", 1, 90, 30)
        recall_df = df[df['Timestamp'] > datetime.utcnow() - timedelta(days=days)]
        recall_df = recall_df[recall_df['Category'].isin(selected_categories)]
        show_completed = st.sidebar.checkbox("Show Completed Items", True)
        debug_mode = st.sidebar.checkbox("Debug Mode", False)
        if debug_mode:
            st.subheader("Raw Data")
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

        if st.button("Summarize Insights"):
            if not recall_df.empty:
                insights = [f"- {row['Insight']} ({row['Category']})" for _, row in recall_df.iterrows() if pd.notnull(row['Insight']) and pd.notnull(row['Category'])]
                prompt = "Summarize these clarity insights by category:\n\n" + "\n".join(insights)
                response = client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role": "system", "content": "You are Clarity Coach."}, {"role": "user", "content": prompt}])
                st.markdown("### Clarity Summary")
                st.write(response.choices[0].message.content)

        st.markdown("---")
        st.header("Completion Metrics")
        df['Week'] = df['Timestamp'].dt.to_period("W").apply(lambda r: r.start_time.date())
        completion_trend = df[df['Status'] == 'Complete'].groupby('Week').size().reset_index(name='Completed')
        fig1 = px.bar(completion_trend, x='Week', y='Completed', title='Weekly Completed Insights')
        st.plotly_chart(fig1, use_container_width=True)
        category_summary = df[df['Status'] == 'Complete'].groupby('Category').size().reset_index(name='Completed')
        fig2 = px.pie(category_summary, names='Category', values='Completed', title='Completed Insights by Category')
        st.plotly_chart(fig2, use_container_width=True)
        total = len(df)
        completed = len(df[df['Status'] == 'Complete'])
        st.metric("Completion Rate", f"{(completed / total * 100):.1f}%" if total > 0 else "0.0%")

    with tabs[2]:
        st.title("Clarity Chat")
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
