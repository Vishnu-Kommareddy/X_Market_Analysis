import os
import platform
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import subprocess

# ─────────────────────────────
# 1️⃣ Cross-Platform Base Path
# ─────────────────────────────
if platform.system() == "Windows":
    BASE_DIR = r"C:\Users\vishn\Downloads\Shift\Programming\code+lab\X_Market_Analysis"
else:
    BASE_DIR = "/mnt/c/Users/vishn/Downloads/Shift/Programming/code+lab/X_Market_Analysis"

DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────
# 2️⃣ Load Environment Variables
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
# 3️⃣ Auto-adjust DB host for WSL (Windows ↔ Linux bridge)
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
# 4️⃣ Load tweets from CSV
# ─────────────────────────────
tweets_file = os.path.join(DATA_DIR, "fed_sentiment_tweets.csv")
if not os.path.exists(tweets_file):
    raise FileNotFoundError(f"❌ Tweets file not found: {tweets_file}")

df = pd.read_csv(tweets_file)
if df.empty:
    print("⚠️ No tweets found to analyze.")
    exit(0)

print(f"📂 Loaded {len(df)} tweets from {tweets_file}")

# ─────────────────────────────
# 5️⃣ Load Hugging Face sentiment model
# ─────────────────────────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print(f"⚙️ Loading Hugging Face model: {MODEL_NAME}")

try:
    sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME, device=-1)
except Exception as e:
    print(f"❌ Model load failed: {e}")
    exit(1)

print(f"🚀 Running sentiment analysis on {len(df)} tweets...")

# ─────────────────────────────
# 6️⃣ Run inference
# ─────────────────────────────
labels, scores = [], []
for tweet_text in tqdm(df["tweet_text"].astype(str), desc="Analyzing"):
    try:
        result = sentiment_pipeline(tweet_text[:512])[0]
        labels.append(result["label"])
        scores.append(result["score"])
    except Exception as e:
        labels.append("ERROR")
        scores.append(None)
        print(f"⚠️ Error processing tweet: {e}")

df["label"] = labels
df["score"] = scores

# ─────────────────────────────
# 7️⃣ Save results locally
# ─────────────────────────────
output_csv = os.path.join(OUTPUT_DIR, "fed_sentiment_hf_scored.csv")
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"📄 Saved Hugging Face scored tweets to {output_csv}")

# ─────────────────────────────
# 8️⃣ Save to PostgreSQL
# ─────────────────────────────
if not all([DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME]):
    print("⚠️ Missing DB credentials — skipping database upload.")
    exit(0)

print(f"🔗 Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}...")
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

create_table_query = """
CREATE TABLE IF NOT EXISTS fed_sentiment_hf_scored (
    tweet_id BIGINT PRIMARY KEY,
    author_id BIGINT,
    tweet_text TEXT NOT NULL,
    created_at TIMESTAMP,
    lang TEXT,
    source TEXT,
    like_count INT,
    retweet_count INT,
    label TEXT,
    score FLOAT
);
"""

with engine.begin() as conn:
    conn.execute(text(create_table_query))

insert_query = """
INSERT INTO fed_sentiment_hf_scored (
    tweet_id, author_id, tweet_text, created_at, lang, source,
    like_count, retweet_count, label, score
)
VALUES (
    :tweet_id, :author_id, :tweet_text, :created_at, :lang, :source,
    :like_count, :retweet_count, :label, :score
)
ON CONFLICT (tweet_id) DO NOTHING;
"""

records = df.to_dict(orient="records")
with engine.begin() as conn:
    conn.execute(text(insert_query), records)

print(f"✅ Inserted or updated {len(records)} records into 'fed_sentiment_hf_scored'.")
