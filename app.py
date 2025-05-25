import streamlit as st
import requests
import json
from datetime import datetime
from openai import OpenAI
import os

# --- SETUP ---
openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"

# --- INIT GPT CLIENT ---
client = OpenAI(api_key=openai_api_key)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Clarity Coach Logger", layout="centered")
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
