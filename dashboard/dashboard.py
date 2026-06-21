# ==============================
# dashboard.py ✅ FINAL CROSS-OS VERSION
# ==============================

import os
import platform
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import streamlit.components.v1 as components
import subprocess

# ╭──────────────────────────────────────────────╮
# │ 1️⃣ BASE PATHS + AUTO HOST DETECTION         │
# ╰──────────────────────────────────────────────╯
if platform.system() == "Windows":
    BASE_DIR = r"C:\Users\vishn\Downloads\Shift\Programming\code+lab\X_Market_Analysis"
    DB_HOST_FALLBACK = "localhost"
else:
    BASE_DIR = "/mnt/c/Users/vishn/Downloads/Shift/Programming/code+lab/X_Market_Analysis"
    try:
        route_output = subprocess.check_output("ip route | grep default", shell=True).decode()
        DB_HOST_FALLBACK = route_output.split("via")[1].split()[0].strip()
    except Exception:
        DB_HOST_FALLBACK = "localhost"

ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# ╭──────────────────────────────────────────────╮
# │ 2️⃣ DATABASE CONNECTION                      │
# ╰──────────────────────────────────────────────╯
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST") or DB_HOST_FALLBACK
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def read_sql_compat(query: str) -> pd.DataFrame:
    """Universal SQL reader compatible with SQLAlchemy 2.x + Pandas 2.x."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
    except Exception as e:
        st.error(f"❌ SQL Read Error: {e}")
        return pd.DataFrame()

# ╭──────────────────────────────────────────────╮
# │ 3️⃣ PAGE CONFIGURATION                       │
# ╰──────────────────────────────────────────────╯
st.set_page_config(page_title="Twitter Sentiment Analytics Dashboard", layout="wide")
st.title("📊 Twitter Sentiment Analytics Dashboard")

# ╭──────────────────────────────────────────────╮
# │ 4️⃣ SIDEBAR FILTERS                          │
# ╰──────────────────────────────────────────────╯
st.sidebar.header("🔍 Filters")

df_dates = read_sql_compat("SELECT created_at FROM fed_sentiment_scored ORDER BY created_at ASC")
if not df_dates.empty:
    min_date = pd.to_datetime(df_dates["created_at"]).min().date()
    max_date = pd.to_datetime(df_dates["created_at"]).max().date()
else:
    min_date = max_date = None

date_filter = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date) if min_date and max_date else None,
    min_value=min_date,
    max_value=max_date,
)

selected_sentiments = st.sidebar.multiselect(
    "TextBlob Sentiment",
    ["positive", "negative", "neutral"],
    ["positive", "negative", "neutral"]
)

selected_hf_labels = st.sidebar.multiselect(
    "Hugging Face Labels",
    ["positive", "negative", "neutral"],
    ["positive", "negative", "neutral"]
)

# ╭──────────────────────────────────────────────╮
# │ 5️⃣ MAIN DASHBOARD TABS                      │
# ╰──────────────────────────────────────────────╯
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 TextBlob Sentiment",
    "🤗 Hugging Face Sentiment",
    "🌐 Network Analysis",
    "🧠 AI Summary"
])

# ╭──────────────────────────────────────────────╮
# │ TAB 1 — TEXTBLOB SENTIMENT                  │
# ╰──────────────────────────────────────────────╯
with tab1:
    st.subheader("💬 TextBlob Sentiment Overview")

    df_textblob = read_sql_compat("SELECT * FROM fed_sentiment_scored")
    if not df_textblob.empty:
        df_textblob["created_at"] = pd.to_datetime(df_textblob["created_at"])
        if date_filter and len(date_filter) == 2:
            start, end = date_filter
            df_textblob = df_textblob[
                (df_textblob["created_at"].dt.date >= start)
                & (df_textblob["created_at"].dt.date <= end)
            ]
        if selected_sentiments:
            df_textblob = df_textblob[df_textblob["sentiment"].isin(selected_sentiments)]

        total = len(df_textblob)
        pos = (df_textblob["sentiment"] == "positive").sum()
        neg = (df_textblob["sentiment"] == "negative").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Tweets", total)
        c2.metric("Positive", pos)
        c3.metric("Negative", neg)

        counts = df_textblob["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        st.plotly_chart(
            px.bar(counts, x="sentiment", y="count", color="sentiment", text="count",
                   title="Sentiment Distribution (TextBlob)"),
            use_container_width=True
        )

        df_textblob["date"] = df_textblob["created_at"].dt.date
        trend = df_textblob.groupby("date")["polarity"].mean().reset_index()
        st.plotly_chart(
            px.line(trend, x="date", y="polarity", title="📈 Average Polarity Trend"),
            use_container_width=True
        )

        st.markdown("### 🏆 Top Positive & Negative Tweets")
        col1, col2 = st.columns(2)
        for _, r in df_textblob.sort_values("polarity", ascending=False).head(5).iterrows():
            col1.info(f"📈 {r['tweet_text']}")
        for _, r in df_textblob.sort_values("polarity", ascending=True).head(5).iterrows():
            col2.error(f"📉 {r['tweet_text']}")
    else:
        st.warning("No TextBlob sentiment data found.")

# ╭──────────────────────────────────────────────╮
# │ TAB 2 — HUGGING FACE SENTIMENT             │
# ╰──────────────────────────────────────────────╯
with tab2:
    st.subheader("🤗 Hugging Face Sentiment (RoBERTa)")

    df_hf = read_sql_compat("SELECT * FROM fed_sentiment_hf_scored")
    if not df_hf.empty:
        df_hf["created_at"] = pd.to_datetime(df_hf["created_at"])
        if date_filter and len(date_filter) == 2:
            s, e = date_filter
            df_hf = df_hf[
                (df_hf["created_at"].dt.date >= s)
                & (df_hf["created_at"].dt.date <= e)
            ]
        if selected_hf_labels:
            df_hf = df_hf[df_hf["label"].isin(selected_hf_labels)]

        total = len(df_hf)
        counts = df_hf["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Tweets", total)
        c2.metric("Positive", int(counts[counts["label"] == "positive"]["count"]) if "positive" in counts["label"].values else 0)
        c3.metric("Negative", int(counts[counts["label"] == "negative"]["count"]) if "negative" in counts["label"].values else 0)

        st.plotly_chart(
            px.bar(counts, x="label", y="count", color="label", text="count",
                   title="Hugging Face Sentiment Distribution"),
            use_container_width=True
        )

        st.markdown("### 🏆 Top Positive & Negative Tweets")
        col1, col2 = st.columns(2)
        for _, r in df_hf[df_hf["label"] == "positive"].sort_values("score", ascending=False).head(5).iterrows():
            col1.info(f"🤗 {r['tweet_text']} (score={r['score']:.2f})")
        for _, r in df_hf[df_hf["label"] == "negative"].sort_values("score", ascending=False).head(5).iterrows():
            col2.error(f"🤗 {r['tweet_text']} (score={r['score']:.2f})")
    else:
        st.warning("No Hugging Face sentiment data found.")

# ╭──────────────────────────────────────────────╮
# │ TAB 3 — NETWORK ANALYSIS VISUALIZATION FIXED │
# ╰──────────────────────────────────────────────╯
with tab3:
    st.subheader("🌐 Twitter Network Graph & Influencers")

    NETWORK_DIR = os.path.join(BASE_DIR, "data", "network")
    html_path = os.path.join(NETWORK_DIR, "network_overview.html")
    nodes_path = os.path.join(NETWORK_DIR, "nodes_metrics.csv")

    # Two-column layout: left = metrics, right = viz
    col1, col2 = st.columns([1.2, 2.8])

    # LEFT: Top Influencers + Hashtags
    with col1:
        if os.path.exists(nodes_path):
            df_nodes = pd.read_csv(nodes_path)
            df_nodes_sorted = df_nodes.sort_values(["pagerank", "degree"], ascending=False)

            top_accounts = df_nodes_sorted[df_nodes_sorted["node_type"].isin(["user", "user_id"])].head(10)
            top_hashtags = df_nodes_sorted[df_nodes_sorted["node_type"] == "hashtag"].head(10)

            st.markdown("### 🌟 Top Influential Accounts")
            for _, r in top_accounts.iterrows():
                st.markdown(f"- **{r['node']}**  \n📊 PR: `{r['pagerank']:.4f}` | Degree: `{r['degree']}`")

            st.markdown("---")
            st.markdown("### 📈 Top Trending Hashtags")
            for _, r in top_hashtags.iterrows():
                st.markdown(f"- **{r['node']}**  \n🏷️ PR: `{r['pagerank']:.4f}` | Degree: `{r['degree']}`")
        else:
            st.warning("⚠️ Run `network_analysis.py` first to generate metrics.")

    # RIGHT: HTML Visualization (Network Graph)
    with col2:
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()

            st.markdown("### 🌐 Interactive Network Graph")
            st.markdown(
                "<div style='border-radius:10px; overflow:hidden; border:1px solid #444;'>",
                unsafe_allow_html=True
            )
            components.html(
                html,
                height=650,
                scrolling=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Network visualization file not found.")

# ╭──────────────────────────────────────────────╮
# │ TAB 4 — AI GENERATED MARKET SUMMARY         │
# ╰──────────────────────────────────────────────╯
with tab4:
    st.subheader("🧠 AI-Generated Market Summary")
    df_summary = read_sql_compat("""
        SELECT summary, generated_at
        FROM tweet_summaries
        ORDER BY generated_at DESC
        LIMIT 1
    """)
    if not df_summary.empty:
        st.info(df_summary.iloc[0]["summary"])
        st.caption(f"🕒 Generated at: {df_summary.iloc[0]['generated_at']}")
    else:
        st.warning("No LLM summaries found. Run llm_summary.py.")
