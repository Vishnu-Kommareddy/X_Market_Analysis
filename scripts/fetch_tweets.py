# ==============================================================
# fetch_tweets.py  ✅ FINAL CROSS-OS + AIRFLOW-READY VERSION
# ==============================================================

import os
import sys
import platform
import tweepy
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Table, Column, Integer, BigInteger, Text, DateTime, MetaData
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv


# ──────────────────────────────────────────────
# 1️⃣ Helper: Timestamped Logging
# ──────────────────────────────────────────────
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# ──────────────────────────────────────────────
# 2️⃣ Cross-Platform Base Directory
# ──────────────────────────────────────────────
if platform.system() == "Windows":
    BASE_DIR = r"C:\Users\vishn\Downloads\Shift\Programming\code+lab\X_Market_Analysis"
else:
    BASE_DIR = "/mnt/c/Users/vishn/Downloads/Shift/Programming/code+lab/X_Market_Analysis"

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 3️⃣ Load Environment Variables
# ──────────────────────────────────────────────
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Auto-resolve DB host for Windows vs WSL
if not DB_HOST or DB_HOST.strip() == "":
    if platform.system() == "Windows":
        DB_HOST = "localhost"
    else:
        DB_HOST = "172.23.240.1"  # Windows host IP for WSL networking

if not BEARER_TOKEN:
    raise ValueError("❌ Missing BEARER_TOKEN in .env file.")


# ──────────────────────────────────────────────
# 4️⃣ Setup Twitter Client
# ──────────────────────────────────────────────
client = tweepy.Client(bearer_token=BEARER_TOKEN)


# ──────────────────────────────────────────────
# 5️⃣ Query Tweets (with optional CLI arg)
# ──────────────────────────────────────────────
query = sys.argv[1] if len(sys.argv) > 1 else "fed rate cut lang:en -is:retweet"
log(f"🔍 Querying Twitter: {query}")

try:
    tweets = client.search_recent_tweets(
        query=query,
        max_results=100,
        tweet_fields=["created_at", "lang", "source", "public_metrics", "author_id"],
    )
except Exception as e:
    print(f"⚠️ Twitter API error: {e}")
    print("⏭️ Skipping fetch_tweets task gracefully (API quota limit reached).")
    exit(0)

if not tweets.data:
    log("⚠️  No tweets found or API limit reached.")
    sys.exit(0)


# ──────────────────────────────────────────────
# 6️⃣ Extract + Transform
# ──────────────────────────────────────────────
data = []
for tweet in tweets.data:
    metrics = tweet.public_metrics or {}
    data.append({
        "tweet_id": tweet.id,
        "author_id": tweet.author_id,
        "tweet_text": tweet.text,
        "created_at": tweet.created_at,
        "lang": tweet.lang,
        "source": tweet.source,
        "like_count": metrics.get("like_count", 0),
        "retweet_count": metrics.get("retweet_count", 0),
        "sentiment": None,  # placeholder for later stages
    })

df = pd.DataFrame(data)


# ──────────────────────────────────────────────
# 7️⃣ Save Locally
# ──────────────────────────────────────────────
csv_path = os.path.join(DATA_DIR, "fed_sentiment_tweets.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")
log(f"📄 Saved tweets to {csv_path}")


# ──────────────────────────────────────────────
# 8️⃣ PostgreSQL Connection + Upsert
# ──────────────────────────────────────────────
if all([DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME]):
    log(f"🔗 Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}...")

    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    metadata = MetaData()

    fed_tweets = Table(
        "fed_tweets", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("tweet_id", BigInteger, unique=True),
        Column("author_id", BigInteger),
        Column("tweet_text", Text, nullable=False),
        Column("created_at", DateTime),
        Column("lang", Text),
        Column("source", Text),
        Column("like_count", Integer),
        Column("retweet_count", Integer),
        Column("sentiment", Text),
        extend_existing=True
    )

    metadata.create_all(engine)

    inserted, skipped = 0, 0
    with engine.begin() as connection:
        for _, row in df.iterrows():
            stmt = (
                insert(fed_tweets)
                .values(**row.to_dict())
                .on_conflict_do_nothing(index_elements=['tweet_id'])
            )
            result = connection.execute(stmt)
            inserted += result.rowcount
            if result.rowcount == 0:
                skipped += 1

    log(f"✅ Tweets inserted: {inserted}, skipped (duplicates): {skipped}")
else:
    log("⚠️  PostgreSQL credentials not found in .env — skipped DB insertion.")


# ──────────────────────────────────────────────
# 9️⃣ End of Script
# ──────────────────────────────────────────────
log("🏁 Fetch tweets script completed successfully.")
