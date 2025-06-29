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

# Timezone
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
def render_category_form(category, clarity_debug):
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
                    st.write("Latest entries:", df.tail(5))

# Main tabs
if openai_ok and sheet_ok:
    tabs = st.tabs(["Clarity Log", "Recall Insights", "Clarity Chat"])

    # Clarity Log Tab
    with tabs[0]:
        st.title("Clarity Coach")
        clarity_debug = st.sidebar.checkbox("Clarity Log Debug Mode", False)
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
            render_category_form(category, clarity_debug)

    # Recall Insights Tab
    with tabs[1]:
        st.title("Recall Insights")
        selected = st.multiselect("Categories", options=categories, default=categories)
        num_entries = st.slider("Entries to display", 5, 200, 50)
        show_completed = st.sidebar.checkbox("Show Completed", False)
        show_timestamps = st.sidebar.checkbox("Show Timestamps", False)
        show_starred = st.sidebar.checkbox("Show Starred Entries Only", False)
        debug_mode = st.sidebar.checkbox("Recall Insight Debug Mode", False)

        df["CreatedAt"] = pd.to_datetime(df["CreatedAt"], errors="coerce", utc=True)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)

        sorted_df = df.sort_values(by="RowIndex", ascending=False).copy()
        filtered_df = sorted_df[
            sorted_df["Category"].isin([c.lower().strip() for c in selected])
        ]

        if not show_completed:
            filtered_df = filtered_df[filtered_df["Status"] != "Complete"]

        if show_starred:
            filtered_df = filtered_df[filtered_df["Priority"].str.lower() == "yes"]

        display_df = filtered_df.head(num_entries)

        if debug_mode:
            st.subheader("🚨 Debug Data")
            st.dataframe(display_df)

        for category in categories:
            cat_lower = category.lower().strip()
            cat_df = display_df[
                display_df["Category"] == cat_lower
            ]

            if cat_df.empty:
                continue

            st.subheader(category.capitalize())

            for idx, row in cat_df.iterrows():
                created_at_str = (
                    row["CreatedAt"].astimezone(local_tz).strftime("%Y-%m-%d %I:%M %p %Z")
                    if pd.notnull(row["CreatedAt"])
                    else "No Log Time"
                )

                label_text = row["Insight"]

                col1, col2 = st.columns([0.85, 0.15])

                with col1:
                    marked = st.checkbox(
                        label_text,
                        key=f"check_{idx}",
                        value=row["Status"] == "Complete",
                    )
                    if show_timestamps:
                        st.markdown(f"**Logged:** {created_at_str}")

                with col2:
                    starred = st.checkbox(
                        "⭐",
                        value=row["Priority"].lower() == "yes",
                        key=f"star_{idx}"
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

    # Clarity Chat Tab
    with tabs[2]:
        st.title("Clarity Chat (AI Coach)")

        chat = st.text_area("Ask Clarity Coach what to focus on:")

        if st.button("Ask"):
            if chat.strip():
                resp = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Clarity Coach, a high-performance AI built to help the user become a millionaire in 6 months. "
                                "You are trained in elite human psychology, decision coaching, and behavior design. "
                                "Your role is not to motivate, but to drive clarity, execution, and accountability across the user’s business and life. "
                                "You cut through distractions, doubts, or emotional spirals quickly. "
                                "You constantly re-anchor the user to their millionaire goal and identity. "
                                "You help the user break big goals into daily tactical moves. "
                                "You ask sharp, smart questions that help the user unlock stuck thinking. "
                                "You provide weekly reviews and structured mindset coaching. "
                                "You operate through five key functions: "
                                "1) Daily Alignment Coach – Define non-negotiables and reset focus. "
                                "2) Strategic Decision Coach – Compare tradeoffs and eliminate distractions. "
                                "3) Identity Shaping Guide – Reinforce the mindset of a 7-figure entrepreneur. "
                                "4) Obstacle Breakdown Coach – Redirect stuck/frustrated energy to focused action. "
                                "5) Weekly Accountability Partner – Track weekly progress, patterns, and corrections. "
                                "Whenever helpful, respond using frameworks, checklists, or pointed questions. "
                                "Avoid comfort or vague encouragement unless explicitly requested. "
                                "Challenge by default. Clarity over complexity. Forward momentum over overthinking. "
                                "Additionally, always help the user figure out which items are most important to focus on, which to delegate, which to hold off on, and which to say no to. "
                                "Provide specific recommendations and rationale."
                            )
                        },
                        {"role": "user", "content": chat},
                    ],
                    temperature=0.2
                )
                st.write(resp.choices[0].message.content)
