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
    header = [h.strip() for h in rows_raw[0]]
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

    # --- LOG TAB (unchanged for brevity) ---

    # --- RECALL TAB ---
    with tabs[1]:
        st.title("🔍 Recall Insights")
        selected_categories = st.multiselect("Select Categories", [d.get("Category", "") for d in data], default=None)
        days = st.slider("Days to look back", 1, 90, 30)

        df = pd.DataFrame(data)
        df.columns = df.columns.str.strip()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])

        if 'Status' not in df.columns:
            df['Status'] = 'Incomplete'
        else:
            df['Status'] = df['Status'].astype(str).str.strip().str.capitalize()

        df['Category'] = df['Category'].astype(str).str.lower().str.strip()

        cutoff = datetime.utcnow() - timedelta(days=days)
        df = df[df['Timestamp'] > cutoff]

        if selected_categories:
            selected_categories = [c.lower() for c in selected_categories]
            df = df[df['Category'].isin(selected_categories)]

        st.subheader("📋 All Rows (After Filtering)")
        st.dataframe(df)

    # --- CHAT TAB ---
    with tabs[2]:
        st.title("💬 Clarity Chat")

        if not df.empty:
            recent_df = df[df['Timestamp'] > datetime.utcnow() - timedelta(days=30)]
            recent_insights = [f"- {row['Insight']} ({row['Category']})" for _, row in recent_df.iterrows() if pd.notnull(row['Insight'])]

            chat_input = st.chat_input("Type your clarity dump, summary request, or question...")
            if chat_input or recent_insights:
                st.chat_message("user").write(chat_input or "Analyze my recent clarity insights")
                system_prompt = "You are Clarity Coach. Help the user gain focus by analyzing the following insights. Identify themes, patterns, and top 80/20 priorities. Provide a short, clear strategic summary."
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chat_input or "\n".join(recent_insights)}
                ]
                response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
                reply = response.choices[0].message.content
                st.chat_message("assistant").write(reply)
            else:
                st.info("You haven't shared any recent brain dumps or insights yet for me to analyze and identify the top 80/20 priorities. Please provide your thoughts, tasks, or notes for today, and I can help determine the key focus areas.")
        else:
            st.warning("No entries found in the last 30 days. Check your sheet or filters.")
