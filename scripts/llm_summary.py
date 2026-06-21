import os
import re
import sys
import platform
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# ╭──────────────────────────────────────────────╮
# │ 1️⃣ Cross-platform paths                     │
# ╰──────────────────────────────────────────────╯
if platform.system() == "Windows":
    BASE_DIR = r"C:\Users\vishn\Downloads\Shift\Programming\code+lab\X_Market_Analysis"
else:
    BASE_DIR = "/mnt/c/Users/vishn/Downloads/Shift/Programming/code+lab/X_Market_Analysis"

ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

SUMMARY_DIR = os.path.join(BASE_DIR, "data", "summaries")
os.makedirs(SUMMARY_DIR, exist_ok=True)
SUMMARY_FILE = os.path.join(SUMMARY_DIR, "latest_summary.txt")

# ╭──────────────────────────────────────────────╮
# │ 2️⃣ Database configuration                   │
# ╰──────────────────────────────────────────────╯
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST") or "localhost"
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Auto-detect Windows host IP for WSL bridge
if platform.system() != "Windows" and DB_HOST in ["localhost", "127.0.0.1", "", None]:
    import subprocess
    try:
        route_output = subprocess.check_output("ip route | grep default", shell=True).decode()
        wsl_host_ip = route_output.split("via")[1].split()[0].strip()
        DB_HOST = wsl_host_ip
        print(f"🌐 Running in WSL — using Windows host IP: {DB_HOST}")
    except Exception as e:
        print(f"⚠️ Could not detect Windows host IP automatically: {e}")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
print(f"🔗 Connected to PostgreSQL at {DB_HOST}:{DB_PORT}")

# ╭──────────────────────────────────────────────╮
# │ 3️⃣ Load tweets from DB                      │
# ╰──────────────────────────────────────────────╯
query = """
SELECT tweet_text, label AS sentiment, created_at
FROM fed_sentiment_hf_scored
ORDER BY created_at DESC
LIMIT 200;
"""

try:
    df = pd.read_sql(query, engine)
    print(f"📥 Loaded {len(df)} tweets for summarization.")
except Exception as e:
    print(f"❌ Error loading tweets: {e}")
    sys.exit(1)

if df.empty:
    print("⚠️ No tweets found to summarize.")
    sys.exit(0)

# ╭──────────────────────────────────────────────╮
# │ 4️⃣ Clean tweet text                         │
# ╰──────────────────────────────────────────────╯
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["tweet_text"].apply(clean_text)
unique_texts = df["clean_text"].drop_duplicates().tolist()

# ╭──────────────────────────────────────────────╮
# │ 5️⃣ Chunking for efficient summarization      │
# ╰──────────────────────────────────────────────╯
def chunk_list(lst, chunk_size=15):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

chunks = list(chunk_list(unique_texts, chunk_size=15))
print(f"✂️ Created {len(chunks)} chunks for summarization.")

# ╭──────────────────────────────────────────────╮
# │ 6️⃣ Initialize OpenRouter / LLM               │
# ╰──────────────────────────────────────────────╯
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")

if not API_KEY or not API_BASE:
    print("❌ Missing OpenRouter credentials (OPENROUTER_API_KEY or OPENAI_API_BASE).")
    sys.exit(1)

try:
    llm = ChatOpenAI(
        model="mistralai/mixtral-8x7b-instruct",  # Or meta-llama/llama-3-8b-instruct
        temperature=0.2,
        max_tokens=500,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
    )
    print("🤖 Loaded LLM successfully.")
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")
    sys.exit(1)

# ╭──────────────────────────────────────────────╮
# │ 7️⃣ Summarize chunks                         │
# ╰──────────────────────────────────────────────╯
chunk_summaries = []
for i, chunk in enumerate(chunks, 1):
    prompt = (
        "Summarize the following tweets in 3–4 sentences. "
        "Focus on overall sentiment and key discussion themes. "
        "Ignore duplicates, links, and hashtags.\n\n"
        + "\n".join(chunk)
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        print(f"✅ Chunk {i}/{len(chunks)} summarized.")
    except Exception as e:
        summary = f"[Error summarizing chunk {i}: {e}]"
        print(f"⚠️ Failed chunk {i}: {e}")

    chunk_summaries.append(summary)

# ╭──────────────────────────────────────────────╮
# │ 8️⃣ Meta-summary (final wrap)                 │
# ╰──────────────────────────────────────────────╯
gold_silver_chunks = [
    c for c in chunk_summaries
    if any(k in c.lower() for k in ["gold", "silver", "xau", "xag", "precious"])
]
other_chunks = [c for c in chunk_summaries if c not in gold_silver_chunks]
ordered_chunks = gold_silver_chunks + other_chunks

meta_prompt = (
    "Summarize the following chunk summaries into one cohesive analytical market wrap. "
    "Include macro themes like Fed policy, interest rates, inflation tone, crypto sentiment, "
    "and gold/silver movements. Keep it concise (4–6 sentences), professional, and neutral.\n\n"
    "Chunk summaries:\n"
    + "\n".join(ordered_chunks)
    + "\n\nFinal Summary:"
)

try:
    final_response = llm.invoke([HumanMessage(content=meta_prompt)])
    final_summary = final_response.content.strip()
except Exception as e:
    print(f"❌ Meta-summary generation failed: {e}")
    sys.exit(1)

# ╭──────────────────────────────────────────────╮
# │ 9️⃣ Clean and display final summary           │
# ╰──────────────────────────────────────────────╯
def clean_summary(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    seen = set()
    unique = []
    for s in sentences:
        s_lower = s.lower()
        if s_lower not in seen:
            unique.append(s)
            seen.add(s_lower)
    return " ".join(unique).strip()

final_summary = clean_summary(final_summary)

print("\n📝 FINAL SUMMARY:\n" + "="*80)
print(final_summary)
print("="*80)

# ╭──────────────────────────────────────────────╮
# │ 🔟 Save locally and in DB                    │
# ╰──────────────────────────────────────────────╯
try:
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(final_summary)
    print(f"💾 Saved summary locally at {SUMMARY_FILE}")
except Exception as e:
    print(f"⚠️ Could not save local summary: {e}")

try:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tweet_summaries (
                id SERIAL PRIMARY KEY,
                summary TEXT,
                generated_at TIMESTAMP DEFAULT NOW()
            );
        """))
        conn.execute(
            text("INSERT INTO tweet_summaries (summary) VALUES (:summary)"),
            {"summary": final_summary}
        )
    print("✅ Summary saved to 'tweet_summaries' table.")
    print(f"🕒 Saved at: {pd.Timestamp.now()}")
except Exception as e:
    print(f"❌ Error saving summary to database: {e}")
    sys.exit(1)

sys.exit(0)
