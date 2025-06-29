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
from googleapiclient.errors import HttpError

# Streamlit Page Config
st.set_page_config(page_title="Clarity Coach", layout="centered")

# API Keys and Webhook URLs
openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"
calendar_webhook_url = "https://hook.us2.make.com/nmd640nukq44ikms638z8w6yavqx1t3f"

# Logging
logging.basicConfig(level=logging.INFO)

# Safe date parsing helper
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

# Cache data with short TTL to ensure fresh data
@st.cache_data(ttl=10)
def load_sheet_data(_attempt=0):
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            json.loads(os.getenv("GOOGLE_SERVICE_KEY")), scope
        )
        gs_client = gspread.authorize(creds)
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
        if df["CreatedAt"].isna().any():
            logging.warning("Rows with invalid CreatedAt detected")
            st.warning("Some entries have invalid timestamps")
        df = df.dropna(subset=["CreatedAt"])
        df["Category"] = df["Category"].astype(str).str.lower().str.strip()
        df["Status"] = df.get("Status", "Incomplete").astype(str).str.strip().str.capitalize()
        df["Priority"] = df.get("Priority", "").astype(str).str.strip()
        df["Device"] = df.get("Device", "").astype(str).str.strip()
        return sheet_ref, df
    except HttpError as e:
        if e.resp.status in [429, 503] and _attempt < 3:
            time.sleep(2 ** _attempt)  # Exponential backoff
            return load_sheet_data(_attempt=_attempt + 1)
        else:
            raise Exception(f"Failed to load sheet after {_attempt + 1} attempts: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to load sheet: {str(e)}")

# OpenAI connectivity
try:
    client = OpenAI(api_key=openai_api_key)
    client.models.list()
    openai_ok = True
except Exception as e:
    openai_ok = False
    st.error("Failed to connect to OpenAI.")
    st.exception(e)

# Connect to Google Sheets
try:
    sheet, df = load_sheet_data()
    sheet_ok = True
except Exception as e:
    sheet_ok = False
    st.error("Failed to connect to Google Sheet.")
    st.exception(e)

# Form for logging entries
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
                    # Post to webhook
                    try:
                        response = requests.post(webhook_url, json=entry)
                        if response.status_code != 200:
                            st.error(f"Webhook failed: {response.text}")
                            logging.error(f"Webhook error: {response.text}")
                    except Exception as e:
                        st.error(f"Webhook error: {str(e)}")
                        logging.error(f"Webhook exception: {str(e)}")
                    # Post to calendar webhook
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
                        logging.warning(f"Calendar webhook error: {str(e)}")
                st.success(f"Logged {len(lines)} insight(s)")
                st.cache_data.clear()  # Clear cache to force refresh
                time.sleep(5)  # Wait for webhook to process
                sheet_new, df_new = load_sheet_data()
                st.write("Latest entries:", df_new.tail(5))
                return sheet_new, df_new  # Return updated sheet and df
    return sheet, df  # Return unchanged sheet and df if no submission

# Main App Tabs
if openai_ok and sheet_ok:
    tabs = st.tabs(["Log Clarity", "Recall Insights", "Clarity Chat"])

    # Log Clarity
    with tabs[0]:
        st.title("Clarity Coach")
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
            sheet, df = render_category_form(category)  # Update sheet and df

    # Recall Insights
    with tabs[1]:
        st.title("Recall Insights")
        selected = st.multiselect(
            "Categories", options=categories, default=categories
        )
        num_entries = st.slider("Entries to display", 5, 200, 50)
        show_completed = st.sidebar.checkbox("Show Completed", True)
        debug_mode = st.sidebar.checkbox("Debug Mode", False)

        sorted_df = df.sort_values(by="CreatedAt", ascending=False).copy()
        filtered_df = sorted_df[
            sorted_df["Category"].isin([c.lower().strip() for c in selected])
        ]

        if not show_completed:
            filtered_df = filtered_df[filtered_df["Status"] != "Complete"]

        display_df = filtered_df.head(num_entries)

        if debug_mode:
            st.subheader("🚨 Debug Data")
            st.dataframe(display_df)

        for idx, row in display_df.iterrows():
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                marked = st.checkbox(
                    f"{row['Insight']} ({row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')})",
                    key=f"check_{idx}",
                    value=row["Status"] == "Complete",
                )
            with col2:
                starred = st.checkbox(
                    "⭐", value=row["Priority"].lower() == "yes", key=f"star_{idx}"
                )

            if marked and row["Status"] != "Complete":
                row_index = df[df["Insight"] == row["Insight"]].index[0] + 2
                sheet.update_cell(row_index, df.columns.get_loc("Status") + 1, "Complete")
                st.success("Marked as complete")

            if starred and row["Priority"].lower() != "yes":
                row_index = df[df["Insight"] == row["Insight"]].index[0] + 2
                sheet.update_cell(row_index, df.columns.get_loc("Priority") + 1, "Yes")
                st.info("Starred")
            elif not starred and row["Priority"].lower() == "yes":
                row_index = df[df["Insight"] == row["Insight"]].index[0] + 2
                sheet.update_cell(row_index, df.columns.get_loc("Priority") + 1, "")
                st.info("Unstarred")

    # Clarity Chat
    with tabs[2]:
        st.title("Clarity Chat (AI Coach)")
        chat = st.text_area("Ask Clarity Coach:")
        if st.button("Ask"):
            if chat.strip():
                resp = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a clarity coach helping users improve their personal and professional life.",
                        },
                        {"role": "user", "content": chat},
                    ],
                )
                st.write(resp.choices[0].message.content)
