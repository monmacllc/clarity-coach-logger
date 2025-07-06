import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pytz
import dateparser
import dateparser.search
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import logging

# Page config
st.set_page_config(page_title="Clarity Coach - Clarity Log", layout="centered")

# Timezone
local_tz = pytz.timezone("US/Pacific")

# API Keys and Webhooks
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"
calendar_webhook_url = "https://hook.us2.make.com/nmd640nukq44ikms638z8w6yavqx1t3f"

# Logging
logging.basicConfig(level=logging.INFO)

# Categories
categories = [
    "ccv",
    "traditional real estate",
    "n&ytg",
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

# Log form per category
def render_category_form(category):
    input_key = f"input_{category}"

    def clear_input():
        st.session_state[input_key] = ""

    with st.expander(category.upper()):
        with st.form(f"{category}_form"):
            input_text = st.text_area(
                f"Insight for {category}",
                height=100,
                key=input_key
            )
            submitted = st.form_submit_button(
                f"Log {category}",
                on_click=clear_input
            )
            if submitted and input_text.strip():
                lines = [
                    s.strip()
                    for chunk in input_text.splitlines()
                    for s in chunk.split(",")
                    if s.strip()
                ]
                for line in lines:
                    start, end, _ = extract_event_info(line)
                    created_at = datetime.utcnow().isoformat(timespec="microseconds")
                    entry = {
                        "timestamp": start,
                        "created_at": created_at,
                        "category": category,
                        "insight": line,
                        "action_step": "",
                        "source": "Clarity Coach",
                        "status": "Incomplete",
                        "priority": "",
                        "device": "Web",
                    }
                    requests.post(webhook_url, json=entry)
                    requests.post(calendar_webhook_url, json={
                        "start": start,
                        "end": end,
                        "summary": line,
                        "category": category,
                        "source": "Clarity Coach",
                    })
                st.success(f"Logged {len(lines)} insight(s)")

# Main page
st.title("📝 Clarity Log")
st.write("Use this page to log your insights.")

for category in categories:
    render_category_form(category)
