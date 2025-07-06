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

# Categories
CATEGORIES = [
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

# Date parsing
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

# Load Google Sheets
def load_sheet_data():
    sheet_ref = gs_client.open("Clarity Capture Log").sheet1
    values = sheet_ref.get_all_values()
    header = [h.strip() for h in values[0]]
    data = []
    for row in values[1:]:
        padded_row = row + [""] * (len(header) - len(row))
        record = dict(zip(header, padded_row))
        data.append(record)
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    df["CreatedAt"] = pd.to_datetime(df["CreatedAt"], errors="coerce", utc=True)
    df["Category"] = df["Category"].astype(str).str.lower().str.strip()
    df["Status"] = df.get("Status", "Incomplete").astype(str).str.strip().str.capitalize()
    df["Priority"] = df.get("Priority", "").astype(str).str.strip()
    return sheet_ref, df

# Connect Sheets
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

# Log form
def render_category_form(category):
    with st.expander(category.upper()):
        with st.form(f"{category}_form"):
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
                time.sleep(2)

# Tabs
if openai_ok and sheet_ok:
    tabs = st.tabs([
        "Clarity Log",
        "Recall Insights",
    ])

    # Clarity Log - Input only
    with tabs[0]:
        st.title("Clarity Coach")
        for category in CATEGORIES:
            render_category_form(category)

    # Recall Insights - Shows everything
    with tabs[1]:
        st.title("Recall Insights")
        selected = st.multiselect(
            "Categories",
            options=[c.upper() for c in CATEGORIES],
            default=[c.upper() for c in CATEGORIES]
        )
        selected_keys = [c.lower().strip() for c in selected]
        num_entries = st.slider("Entries to display", 5, 200, 50)
        show_completed = st.sidebar.checkbox("Show Completed", False)
        show_starred = st.sidebar.checkbox("Show Starred Entries Only", False)

        filtered_df = df[
            df["Category"].isin(selected_keys)
        ]
        if not show_completed:
            filtered_df = filtered_df[filtered_df["Status"] != "Complete"]
        if show_starred:
            filtered_df = filtered_df[filtered_df["Priority"].str.lower() == "yes"]

        display_df = filtered_df.sort_values(by="CreatedAt", ascending=False).head(num_entries)

        for category in CATEGORIES:
            cat_df = display_df[display_df["Category"] == category]
            if cat_df.empty:
                continue
            st.subheader(category.upper())
            for idx, row in cat_df.iterrows():
                st.markdown(f"- {row['Insight']} ({row['Status']})")

            # Clarity Chat Tab
    with tabs[2]:
        st.title("Clarity Chat (AI Coach)")

        # Collect latest Recall Insights context
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
            for idx, row in combined_recent.iterrows():
                insights_context += f"- {row['Category'].capitalize()}: {row['Insight']}\n"

        st.markdown("**💡 Quick Prompts:**")
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
Your role is not to motivate, but to drive clarity, execution, and accountability across the user’s business and life.
You cut through distractions, doubts, or emotional spirals quickly.
You constantly re-anchor the user to their millionaire goal and identity.
You help the user break big goals into daily tactical moves.
You ask sharp, smart questions that help the user unlock stuck thinking.
You provide weekly reviews and structured mindset coaching.
You operate through five key functions:
1) Daily Alignment Coach – Define non-negotiables and reset focus.
2) Strategic Decision Coach – Compare tradeoffs and eliminate distractions.
3) Identity Shaping Guide – Reinforce the mindset of a 7-figure entrepreneur.
4) Obstacle Breakdown Coach – Redirect stuck/frustrated energy to focused action.
5) Weekly Accountability Partner – Track weekly progress, patterns, and corrections.
Whenever helpful, respond using frameworks, checklists, or pointed questions.
Avoid comfort or vague encouragement unless explicitly requested.
Challenge by default. Clarity over complexity. Forward momentum over overthinking.
Additionally, always help the user figure out which items are most important to focus on, which to delegate, which to hold off on, and which to say no to.
Provide specific recommendations and rationale.
"""
                        },
                        {
                            "role": "user",
                            "content": insights_context + "\n\n" + prompt_text
                        },
                    ],
                    temperature=0.2
                )
                st.write(resp.choices[0].message.content)

        # Quick prompt buttons
        if col1.button("What are the top 3 moves I need to make today?"):
            run_clarity_chat("What are the top 3 moves I need to make today?")

        if col2.button("I'm stuck—help me refocus fast."):
            run_clarity_chat("I'm stuck—help me refocus fast.")

        if col3.button("What’s the clearest way to reach my income goal?"):
            run_clarity_chat("What’s the clearest way to reach my income goal?")

        st.markdown("---")
        chat = st.text_area("Or ask your own question:")

        if st.button("Ask"):
            if chat.strip():
                run_clarity_chat(chat)

# Insight Dashboard Tab
with tabs[3]:
    st.title("📊 Insights Dashboard")

    try:
        # Prepare timestamps
        df["CreatedAt"] = pd.to_datetime(df["CreatedAt"], errors="coerce", utc=True)
        df_filtered = df.copy()

        # Calculate DaysAgo
        df_filtered["DaysAgo"] = df_filtered["CreatedAt"].apply(
            lambda d: (pd.Timestamp.utcnow() - d).days
        )

        # Assign timeframes
        def bucket_label(days_ago):
            if days_ago <= 7:
                return "Last 7 Days"
            elif days_ago <= 14:
                return "Last 14 Days"
            elif days_ago <= 21:
                return "Last 21 Days"
            elif days_ago <= 30:
                return "Last 30 Days"
            else:
                return None

        df_filtered["Timeframe"] = df_filtered["DaysAgo"].apply(bucket_label)
        df_filtered = df_filtered[df_filtered["Timeframe"].notnull()]
        df_filtered["Status"] = df_filtered["Status"].str.strip().str.capitalize()

        # Aggregate counts per timeframe
        all_timeframes = ["Last 7 Days", "Last 14 Days", "Last 21 Days", "Last 30 Days"]
        all_statuses = ["Complete", "Incomplete"]

        entries_per_timeframe = (
            df_filtered.groupby(["Timeframe", "Status"])
            .size()
            .reset_index(name="Count")
        )

        idx = pd.MultiIndex.from_product(
            [all_timeframes, all_statuses],
            names=["Timeframe", "Status"]
        )

        entries_per_timeframe = (
            entries_per_timeframe
            .set_index(["Timeframe", "Status"])
            .reindex(idx, fill_value=0)
            .reset_index()
        )

        # Timeframe checkboxes with simplified labels
        st.markdown("### Entries by Disjoint Timeframes")
        col1, col2, col3 = st.columns(3)
        show_14 = col1.checkbox("Include Last 14 Days")
        show_21 = col2.checkbox("Include Last 21 Days")
        show_30 = col3.checkbox("Include Last 30 Days")

        selected_buckets = ["Last 7 Days"]
        if show_14:
            selected_buckets.append("Last 14 Days")
        if show_21:
            selected_buckets.append("Last 21 Days")
        if show_30:
            selected_buckets.append("Last 30 Days")

        entries_per_timeframe["Show"] = entries_per_timeframe["Timeframe"].isin(selected_buckets)
        entries_per_timeframe["DisplayCount"] = entries_per_timeframe.apply(
            lambda row: row["Count"] if row["Show"] else 0,
            axis=1
        )
        entries_per_timeframe["label_text"] = entries_per_timeframe["DisplayCount"].apply(
            lambda x: str(x) if x > 0 else ""
        )
        entries_per_timeframe["Timeframe"] = pd.Categorical(
            entries_per_timeframe["Timeframe"],
            categories=all_timeframes,
            ordered=True
        )

        if entries_per_timeframe["DisplayCount"].sum() == 0:
            st.info("No entries to display.")
        else:
            base = alt.Chart(entries_per_timeframe).encode(
                x=alt.X("Timeframe:N", sort=all_timeframes),
                y=alt.Y("DisplayCount:Q", title="Number of Entries"),
                color=alt.Color("Status:N"),
                tooltip=["Timeframe", "Status", "DisplayCount"]
            )
            bars = base.mark_bar()
            text_inside = base.transform_filter(
                alt.datum.DisplayCount >= 10
            ).mark_text(
                align="center",
                dy=5,
                color="black"
            ).encode(text="label_text:N")
            text_above = base.transform_filter(
                alt.datum.DisplayCount < 10
            ).mark_text(
                align="center",
                dy=-10,
                color="black"
            ).encode(text="label_text:N")
            chart = (bars + text_inside + text_above).properties(height=400)
            st.altair_chart(chart, use_container_width=True)

        # Completed entries by category
        with st.expander("🥧 Completed Entries by Category (Last 30 Days)"):
            categories_order = [
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

            cutoff_date = pd.Timestamp.utcnow() - pd.Timedelta(days=30)
            completed_30 = df[
                (df["Status"] == "Complete") & (df["CreatedAt"] >= cutoff_date)
            ]

            counts = (
                completed_30.groupby("Category")
                .size()
                .reset_index(name="CompletedCount")
            )

            all_cats_df = pd.DataFrame({"Category": categories_order})
            merged_counts = pd.merge(all_cats_df, counts, on="Category", how="left").fillna(0)
            merged_counts["CompletedCount"] = merged_counts["CompletedCount"].astype(int)

            if merged_counts["CompletedCount"].sum() == 0:
                st.info("No completed entries in the past 30 days.")
            else:
                pie = alt.Chart(merged_counts).mark_arc(innerRadius=40).encode(
                    theta=alt.Theta("CompletedCount:Q"),
                    color=alt.Color("Category:N", sort=categories_order),
                    tooltip=["Category", "CompletedCount"]
                ).properties(height=400)
                st.altair_chart(pie, use_container_width=True)

                bar = alt.Chart(merged_counts).mark_bar().encode(
                    x=alt.X("CompletedCount:Q", title="Completed Entries"),
                    y=alt.Y("Category:N", sort="-x"),
                    tooltip=["Category", "CompletedCount"]
                ).properties(height=400, title="Completed Entries per Category")
                st.altair_chart(bar, use_container_width=True)

    except Exception as e:
        st.error("⚠️ An error occurred while rendering the Insights Dashboard.")
        st.exception(e)

