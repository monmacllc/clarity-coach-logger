import streamlit as st
import json
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI

# Page config
st.set_page_config(page_title="Clarity Coach - Clarity Chat", layout="centered")

# API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

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

# Load data
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    json.loads(os.getenv("GOOGLE_SERVICE_KEY")), scope
)
gs_client = gspread.authorize(creds)
sheet = gs_client.open("Clarity Capture Log").sheet1

values = sheet.get_all_values()
header = [h.strip() for h in values[0]]
data = []
for row in values[1:]:
    padded_row = row + [""] * (len(header) - len(row))
    record = dict(zip(header, padded_row))
    data.append(record)
df = pd.DataFrame(data)
df.columns = df.columns.str.strip()

df["CreatedAt"] = pd.to_datetime(df["CreatedAt"], errors="coerce", utc=True)
df["Status"] = df.get("Status", "Incomplete").astype(str).str.strip().str.capitalize()
df["Priority"] = df.get("Priority", "").astype(str).str.strip()

# Recent context
cutoff_30 = pd.Timestamp.utcnow() - pd.Timedelta(days=30)
recent_incomplete = df[
    (df["Status"] != "Complete") &
    (df["CreatedAt"] >= cutoff_30)
]
recent_starred = df[
    (df["Priority"].str.lower() == "yes") &
    (df["CreatedAt"] >= cutoff_30)
]
combined_recent = pd.concat([recent_incomplete, recent_starred]).drop_duplicates()

insights_context = ""
if not combined_recent.empty:
    insights_context += "Here are my current incomplete or important entries:\n"
    for _, row in combined_recent.iterrows():
        insights_context += f"- {row['Category'].capitalize()}: {row['Insight']}\n"

# Page UI
st.title("💬 Clarity Chat (AI Coach)")
st.markdown("Use quick prompts or ask your own question:")

col1, col2, col3 = st.columns(3)

def run_clarity_chat(prompt_text):
    with st.spinner("Analyzing..."):
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """
You are Clarity Coach, a high-performance AI built to help the user become a millionaire in 6 months.
You are trained in elite human psychology, decision coaching, and behavior design.
Challenge by default. Clarity over complexity. Forward momentum over overthinking.
"""
                },
                {"role": "user", "content": insights_context + "\n\n" + prompt_text},
            ],
            temperature=0.2
        )
        st.write(resp.choices[0].message.content)

if col1.button("What are the top 3 moves I need to make today?"):
    run_clarity_chat("What are the top 3 moves I need to make today?")

if col2.button("I'm stuck—help me refocus fast."):
    run_clarity_chat("I'm stuck—help me refocus fast.")

if col3.button("What's the clearest way to reach my income goal?"):
    run_clarity_chat("What's the clearest way to reach my income goal?")

st.markdown("---")
chat = st.text_area("Or ask your own question:")

if st.button("Ask"):
    if chat.strip():
        run_clarity_chat(chat)

