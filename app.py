import streamlit as st
import requests
import json
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as dtparser
from openai import OpenAI
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- SETUP ---
openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"

# --- INIT GPT CLIENT ---
client = OpenAI(api_key=openai_api_key)

# --- GOOGLE SHEETS SETUP ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
service_key_json = os.getenv("GOOGLE_SERVICE_KEY")
creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(service_key_json), scope)
gs_client = gspread.authorize(creds)
sheet = gs_client.open("Clarity Capture Log").sheet1
rows = sheet.get_all_records()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Clarity Coach Logger", layout="centered")
tabs = st.tabs(["🚀 Log Clarity", "🔍 Recall Insights", "💬 Clarity Chat"])

# --- LOG TAB ---
with tabs[0]:
    st.title("🧠 Clarity Coach Logger")
    st.write("Paste your brain dump or clarity notes below. We'll auto-organize and log them.")

    user_input = st.text_area("Clarity Input", height=200)

    if st.button("🚀 Log Insights"):
        with st.spinner("Thinking like Clarity Coach..."):
            system_prompt = """
You are Clarity Coach. Parse brain dumps into structured JSON entries. For each bullet point, return:
- timestamp (ISO format)
- category
- insight
- action_step (optional)
- source = Clarity Coach
Return a list of objects.
"""
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                log_status = []
                for entry in data:
                    entry['timestamp'] = datetime.utcnow().isoformat()
                    r = requests.post(webhook_url, json=entry)
                    log_status.append(f"✅ {entry['category']}: {entry['insight']} | Status: {r.status_code}")
                st.success("Entries logged successfully!")
                st.write("\n".join(log_status))
            except Exception as e:
                st.error("Failed to parse GPT output or send to Make webhook.")
                st.code(content)
                st.exception(e)

# --- RECALL TAB ---
with tabs[1]:
    st.title("🔍 Recall Insights")
    category_options = [
        "ccv", "traditional real estate", "co living", "finances", "body", "mind", "spirit",
        "family", "kids", "wife", "relationships", "quality of life", "fun", "giving back", "stressors"
    ]
    selected_categories = st.multiselect("Select Categories", category_options, default=category_options[:3])
    days = st.slider("Days to look back", 1, 90, 7)

    if st.button("🧠 Summarize Insights"):
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        insights = []
        filtered_rows = []

        for r in rows:
            ts = r.get('Timestamp')
            try:
                if ts:
                    if isinstance(ts, int):
                        ts_dt = datetime.utcfromtimestamp(ts)
                    else:
                        ts_dt = dtparser(str(ts))

                    raw_cat = r['Category'].lower().strip()
                    category_map = {
                        "ccv (main business)": "ccv",
                        "traditional real estate": "traditional real estate",
                        "co-living": "co living",
                        "family relationships - wife": "wife",
                        "family relationships - kids": "kids",
                        "family relationships - extended family": "family"
                    }
                    cat = category_map.get(raw_cat, raw_cat)

                    if any(cat.startswith(sel.lower()) for sel in selected_categories) and ts_dt > cutoff:
                        insights.append(f"- {r['Insight']} ({ts_dt.date()})")
                        filtered_rows.append(r)
            except Exception:
                continue

        if not insights:
            st.info("No entries found for those filters.")
        else:
            prompt = f"Summarize these clarity insights from the last {days} days under categories {', '.join(selected_categories)}:\n\n" + "\n".join(insights)
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are Clarity Coach. Return a structured, helpful summary."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.success("🧠 Clarity Summary:")
            st.write(response.choices[0].message.content)

# --- CHAT TAB ---
with tabs[2]:
    st.title("💬 Clarity Chat")
    chat_input = st.chat_input("Type your clarity dump, summary request, or question...")
    if chat_input:
        st.chat_message("user").write(chat_input)
        system_prompt = "You are Clarity Coach. Decide whether the input is a brain dump to log or a recall request. If it's a dump, return a JSON list of insights with timestamp, category, insight, action_step (optional), and source = Clarity Coach. If it's a recall request, summarize past entries. Use ONLY available data."
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_input}
            ]
        )
        reply = response.choices[0].message.content
        st.chat_message("assistant").write(reply)

        try:
            entries = json.loads(reply)
            log_status = []
            for entry in entries:
                entry['timestamp'] = datetime.utcnow().isoformat()
                r = requests.post(webhook_url, json=entry)
                log_status.append(f"✅ {entry['category']}: {entry['insight']} | Status: {r.status_code}")
            st.success("Auto-logged all entries.")
        except:
            pass
