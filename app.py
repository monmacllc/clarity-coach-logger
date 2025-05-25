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
st.set_page_config(page_title="Clarity Coach Logger", layout="centered")
openai_api_key = os.getenv("OPENAI_API_KEY")
webhook_url = "https://hook.us2.make.com/lagvg0ooxpjvgcftceuqgllovbbr8h42"

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

    # Test Google Sheets access directly
    test_values = sheet.get_all_values()
    st.success(f"✅ Successfully accessed Google Sheet. First row: {test_values[0]}")

    # Load data
    rows_raw = test_values
    header = rows_raw[0]
    data = [dict(zip(header, row + [''] * (len(header) - len(row)))) for row in rows_raw[1:] if any(row)]
    sheet_ok = True

except Exception as e:
    sheet_ok = False
    st.error("❌ Failed to connect to Google Sheet. Make sure the sheet is shared with your service account.")
    st.exception(e)


# --- STREAMLIT TABS ---
if openai_ok and sheet_ok:
    tabs = st.tabs(["🚀 Log Clarity", "🔍 Recall Insights", "💬 Clarity Chat"])

    # --- LOG TAB ---
    with tabs[0]:
        st.title("🧠 Clarity Coach Logger")
        st.write("Enter your insights directly by category. Each form below logs to your sheet.")

        categories = [
            "ccv", "traditional real estate", "stressors", "co living", "finances",
            "body mind spirit", "wife", "kids", "family", "quality of life",
            "fun", "giving back", "misc"
        ]

        for category in categories:
            with st.expander(category.upper()):
                with st.form(key=f"form_{category}"):
                    input_text = st.text_area(f"Insight for {category}", key=f"input_{category}", height=100)
                    submitted = st.form_submit_button(f"Log {category} Insight")
                    if submitted and input_text.strip():
                        lines = [s.strip() for chunk in input_text.splitlines() for s in chunk.split(',') if s.strip()]
                        for line in lines:
                            entry = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "category": category,
                                "insight": line,
                                "action_step": "",
                                "source": "Clarity Coach"
                            }
                            r = requests.post(webhook_url, json=entry)
                        st.success(f"✅ Logged {len(lines)} insight(s) under {category}")

    # --- RECALL TAB ---
    with tabs[1]:
        st.title("🔍 Recall Insights")
        selected_categories = st.multiselect("Select Categories", categories, default=categories[:3])
        days = st.slider("Days to look back", 1, 90, 7)

        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        df['Status'] = df['Status'].astype(str).str.strip().str.capitalize()
        df['Category'] = df['Category'].astype(str).str.lower().str.strip()
        selected_categories = [c.lower() for c in selected_categories]

        cutoff = datetime.utcnow() - timedelta(days=days)
        df = df[df['Timestamp'] > cutoff]
        df = df[df['Category'].isin(selected_categories)]

        show_completed = st.sidebar.checkbox("Show Completed Items", value=False)
        debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

        if debug_mode:
            st.subheader("📋 All Rows (Before Filtering)")
            st.dataframe(pd.DataFrame(data))

        if show_completed:
            filtered_df = df[df['Status'] == 'Complete']
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
            st.write("🔍 Filtered rows count:", len(filtered_df))
            if not filtered_df.empty:
                if debug_mode:
                    st.subheader("📋 Filtered Data Before Summary")
                    st.dataframe(filtered_df)
                insight_texts = [f"- {row['Insight']} ({row['Category']})" for _, row in filtered_df.iterrows() if pd.notnull(row['Insight']) and pd.notnull(row['Category'])]
                if insight_texts:
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
                    st.info("No valid insights to summarize.")
            else:
                st.info("No insights available to summarize. Try adjusting filters.")

        # --- KPI SECTION ---
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

        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        df['Category'] = df['Category'].astype(str).str.lower().str.strip()
        df['Insight'] = df['Insight'].astype(str)

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
