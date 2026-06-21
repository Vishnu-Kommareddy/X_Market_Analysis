# =====================================================
# x_market_etl_dag.py  ✅ FINAL WORKING DAG
# =====================================================

import os
import platform
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# ───────────────────────────────────────────────
# 1️⃣ Cross-platform path setup
# ───────────────────────────────────────────────
if platform.system() == "Windows":
    PROJECT_DIR = r"C:\Users\vishn\Downloads\Shift\Programming\code+lab\X_Market_Analysis"
else:
    PROJECT_DIR = "/mnt/c/Users/vishn/Downloads/Shift/Programming/code+lab/X_Market_Analysis"

SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")

# ───────────────────────────────────────────────
# 2️⃣ Airflow default args
# ───────────────────────────────────────────────
default_args = {
    "owner": "vishn",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

# ───────────────────────────────────────────────
# 3️⃣ DAG definition
# ───────────────────────────────────────────────
with DAG(
    dag_id="x_market_etl_dag",
    default_args=default_args,
    description="End-to-end ETL + Sentiment + Network + LLM Summary",
    schedule_interval=None,  # Run manually
    start_date=datetime(2025, 10, 30),
    catchup=False,
    tags=["x_market", "ETL", "sentiment", "pipeline"],
) as dag:

    # Extract tweets
    fetch_tweets = BashOperator(
        task_id="fetch_tweets",
        bash_command=f"python '{os.path.join(SCRIPTS_DIR, 'fetch_tweets.py')}'"
    )

    # TextBlob sentiment
    textblob_sentiment = BashOperator(
        task_id="textblob_sentiment",
        bash_command=f"python '{os.path.join(SCRIPTS_DIR, 'textblob_sentiment.py')}'"
    )

    # Hugging Face sentiment
    hf_sentiment = BashOperator(
        task_id="huggingface_sentiment",
        bash_command=f"python '{os.path.join(SCRIPTS_DIR, 'huggingface_sentiment.py')}'"
    )

    # Network analysis
    network_analysis = BashOperator(
        task_id="network_analysis",
        bash_command=f"python '{os.path.join(SCRIPTS_DIR, 'network_analysis.py')}'"
    )

    # LLM summary generation
    llm_summary = BashOperator(
        task_id="llm_summary",
        bash_command=f"python '{os.path.join(SCRIPTS_DIR, 'llm_summary.py')}'"
    )

    # Task dependencies
    fetch_tweets >> textblob_sentiment >> hf_sentiment >> network_analysis >> llm_summary
