import os
import platform
import pandas as pd
from textblob import TextBlob
from sqlalchemy import create_engine
from dotenv import load_dotenv
import re, subprocess

# ─────────────────────────────
# 1️⃣ Base path (Windows ↔️ WSL aware)
# ─────────────────────────────
if platform.system() == "Windows":
    BASE_DIR = r"C:\Users\vishn\Downloads\Shift\Programming\code+lab\X_Market_Analysis"
else:
    BASE_DIR = "/mnt/c/Users/vishn/Downloads/Shift/Programming/code+lab/X_Market_Analysis"

DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────
# 2️⃣ Load environment variables
# ─────────────────────────────
env_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(env_path):
    raise FileNotFoundError(f"❌ .env file not found at: {env_path}")

load_dotenv(env_path)

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# ─────────────────────────────
# 3️⃣ Auto-adjust DB host (for WSL bridge)
# ─────────────────────────────
if platform.system() != "Windows":
    try:
        route_output = subprocess.check_output("ip route | grep default", shell=True).decode()
        wsl_host_ip = route_output.split("via")[1].split()[0].strip()
        DB_HOST = DB_HOST or wsl_host_ip
        print(f"🌐 Running in WSL — using Windows host IP: {DB_HOST}")
    except Exception as e:
        print(f"⚠️ Could not auto-detect Windows host IP: {e}")
else:
    DB_HOST = DB_HOST or "localhost"

# ─────────────────────────────
# 4️⃣ PostgreSQL connection
# ─────────────────────────────
engine = None
if all([DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME]):
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"🔗 Connected to PostgreSQL at {DB_HOST}:{DB_PORT}")
else:
    print("⚠️ Missing DB credentials — skipping database operations.")

# ─────────────────────────────
# 5️⃣ Load input tweets CSV
# ─────────────────────────────
input_file = os.path.join(DATA_DIR, "fed_sentiment_tweets.csv")
if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ Input CSV not found: {input_file}")

tweets_df = pd.read_csv(input_file, encoding="utf-8")
print(f"📂 Loaded {len(tweets_df)} tweets from {input_file}")

# ─────────────────────────────
# 6️⃣ Fetch existing tweet IDs
# ─────────────────────────────
existing_ids = set()
if engine:
    try:
        existing_df = pd.read_sql("SELECT tweet_id FROM fed_sentiment_scored", engine)
        existing_ids = set(existing_df["tweet_id"])
        print(f"🧮 Found {len(existing_ids)} existing analyzed tweets.")
    except Exception:
        print("ℹ️ Table 'fed_sentiment_scored' not found — starting fresh.")

new_tweets_df = tweets_df[~tweets_df["tweet_id"].isin(existing_ids)]
if new_tweets_df.empty:
    print("✅ No new tweets to analyze.")
    exit()

# ─────────────────────────────
# 7️⃣ Sentiment analysis
# ─────────────────────────────
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.05:
        return "positive"
    elif polarity < -0.05:
        return "negative"
    else:
        return "neutral"

print("🧠 Performing TextBlob sentiment analysis...")
new_tweets_df["polarity"] = new_tweets_df["tweet_text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
new_tweets_df["sentiment"] = new_tweets_df["tweet_text"].apply(get_sentiment)

# ─────────────────────────────
# 8️⃣ Save results to database
# ─────────────────────────────
if engine:
    try:
        with engine.connect() as conn:
            raw_conn = conn.connection
            new_tweets_df.to_sql("fed_sentiment_scored", con=raw_conn, if_exists="append", index=False)
        print(f"✅ {len(new_tweets_df)} new tweets appended to 'fed_sentiment_scored'.")
    except Exception as e:
        print(f"❌ Database insertion failed: {e}")
else:
    print("⚠️ Skipped DB insertion due to missing connection.")

# ─────────────────────────────
# 9️⃣ Local CSV backup
# ─────────────────────────────
output_file = os.path.join(OUTPUT_DIR, "fed_sentiment_with_scores.csv")
header = not os.path.exists(output_file)
new_tweets_df.to_csv(output_file, mode="a", index=False, header=header, encoding="utf-8")
print(f"📁 Backup saved to: {output_file}")
