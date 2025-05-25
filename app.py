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

    category_options = ["ccv", "traditional real estate", "co living", "finances", "body", "mind", "spirit", "family", "kids", "wife", "relationships", "quality of life", "fun", "giving back", "stressors", "communication", "testing", "performance review", "appointments", "task", "project management", "travel planning", "morning routine", "preparation"]
    approved_categories = ", ".join(category_options)

    if st.button("🚀 Log Insights"):
        with st.spinner("Thinking like Clarity Coach..."):
            system_prompt = f"""
You are Clarity Coach. Parse brain dumps into structured JSON entries. Each entry must use one of the following categories: {approved_categories}.
- If the text matches multiple categories, choose the most precise match.
- If it doesn't clearly fit, assign to 'other'.
Return a list of objects with:
- timestamp (ISO format)
- category
- insight
- action_step (optional)
- source = Clarity Coach
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
                    if entry['category'].lower() not in [c.lower() for c in category_options]:
                        entry['category'] = 'other'
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
    selected_categories = st.multiselect("Select Categories", category_options, default=category_options[:3])
    days = st.slider("Days to look back", 1, 90, 7)

    df = pd.DataFrame(rows)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    df = df[df['Category'].isin(category_options)]

    cutoff = datetime.utcnow() - timedelta(days=days)
    df = df[df['Timestamp'] > cutoff]
    df = df[df['Category'].isin(selected_categories)]

    show_completed = st.sidebar.checkbox("Show Completed Items", value=False)

    if show_completed:
        filtered_df = df[df['Status'] == 'Complete']
        try:
            min_date = filtered_df['Timestamp'].min().date()
        except:
            min_date = date.today()
        try:
            max_date = filtered_df['Timestamp'].max().date()
        except:
            max_date = date.today()

        selected_category = st.sidebar.selectbox("Filter by Category", ["All"] + sorted(filtered_df['Category'].unique().tolist()))
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['Category'] == selected_category]

        start_date = st.sidebar.date_input("Start Date", value=min_date)
        end_date = st.sidebar.date_input("End Date", value=max_date)
        mask = (filtered_df['Timestamp'].dt.date >= start_date) & (filtered_df['Timestamp'].dt.date <= end_date)
        filtered_df = filtered_df[mask]
    else:
        filtered_df = df[df['Status'] != 'Complete']

    grouped = filtered_df.groupby('Category')
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
        if not filtered_df.empty:
            insight_texts = [f"- {row['Insight']} ({row['Category']})" for _, row in filtered_df.iterrows()]
            prompt = "Summarize these clarity insights by category:\n\n" + "\n".join(insight_texts)
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are Clarity Coach. Return a structured, insightful summary by category."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.markdown("### 🧠 Clarity Summary")
            st.write(response.choices[0].message.content)
        else:
            st.info("No insights available to summarize. Try adjusting filters.")

    # --- KPI Section ---
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
