import os
import re
import itertools
import platform
import pandas as pd
import networkx as nx
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ╭──────────────────────────────────────────────────────────────╮
# │ 1️⃣ Cross-platform base paths                                 │
# ╰──────────────────────────────────────────────────────────────╯
if platform.system() == "Windows":
    BASE_DIR = r"C:\Users\vishn\Downloads\Shift\Programming\code+lab\X_Market_Analysis"
else:
    BASE_DIR = "/mnt/c/Users/vishn/Downloads/Shift/Programming/code+lab/X_Market_Analysis"

DATA_DIR = os.path.join(BASE_DIR, "data")
NETWORK_DIR = os.path.join(DATA_DIR, "network")
os.makedirs(NETWORK_DIR, exist_ok=True)

# ╭──────────────────────────────────────────────────────────────╮
# │ 2️⃣ Environment + PostgreSQL setup                            │
# ╰──────────────────────────────────────────────────────────────╯
env_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(env_path):
    raise FileNotFoundError(f"❌ .env not found at {env_path}")

load_dotenv(env_path)

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Auto-detect Windows host IP for WSL bridge
if platform.system() != "Windows" and (not DB_HOST or DB_HOST in ["localhost", "127.0.0.1"]):
    import subprocess
    try:
        route_output = subprocess.check_output("ip route | grep default", shell=True).decode()
        wsl_host_ip = route_output.split("via")[1].split()[0].strip()
        DB_HOST = wsl_host_ip
        print(f"🌐 Running in WSL — using Windows host IP: {DB_HOST}")
    except Exception as e:
        print(f"⚠️ Could not detect Windows host IP automatically: {e}")

ENGINE = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
print(f"🔗 Connected to PostgreSQL at {DB_HOST}:{DB_PORT}")

# ╭──────────────────────────────────────────────────────────────╮
# │ 3️⃣ Load tweet data (with sentiment)                          │
# ╰──────────────────────────────────────────────────────────────╯
QUERY = """
SELECT tweet_id, author_id, tweet_text, created_at, sentiment, polarity
FROM fed_sentiment_scored
WHERE tweet_text IS NOT NULL
"""
df = pd.read_sql(text(QUERY), ENGINE)

if df.empty:
    print("⚠️ No tweets found in fed_sentiment_scored — run ETL + sentiment scripts first.")
    raise SystemExit

print(f"📂 Loaded {len(df)} tweets from database")

# ╭──────────────────────────────────────────────────────────────╮
# │ 4️⃣ Regex extraction for mentions & hashtags                   │
# ╰──────────────────────────────────────────────────────────────╯
MENTION_REGEX = re.compile(r"@([A-Za-z0-9_]{1,15})")
HASHTAG_REGEX = re.compile(r"#([A-Za-z0-9_]+)")

def extract_mentions(text: str):
    return MENTION_REGEX.findall(text or "")

def extract_hashtags(text: str):
    return [h.lower() for h in HASHTAG_REGEX.findall(text or "")]

df["mentions"] = df["tweet_text"].apply(extract_mentions)
df["hashtags"] = df["tweet_text"].apply(extract_hashtags)

# ╭──────────────────────────────────────────────────────────────╮
# │ 5️⃣ Mention network (directed)                                │
# ╰──────────────────────────────────────────────────────────────╯
mention_edges = []
for _, row in df.iterrows():
    src = f"user:{row['author_id']}"
    for handle in row["mentions"]:
        dst = f"@{handle}"
        mention_edges.append((src, dst))

mention_df = pd.DataFrame(mention_edges, columns=["source", "target"])
if not mention_df.empty:
    mention_df["weight"] = 1
    mention_df = mention_df.groupby(["source", "target"], as_index=False)["weight"].sum()

Gm = nx.DiGraph()
if not mention_df.empty:
    for _, r in mention_df.iterrows():
        Gm.add_edge(r["source"], r["target"], weight=int(r["weight"]))

# ╭──────────────────────────────────────────────────────────────╮
# │ 6️⃣ Hashtag co-occurrence network (undirected)                │
# ╰──────────────────────────────────────────────────────────────╯
hashtag_edges = []
for tags in df["hashtags"]:
    tags = list(dict.fromkeys(tags))  # remove duplicates within tweet
    if len(tags) >= 2:
        for a, b in itertools.combinations(sorted(tags), 2):
            hashtag_edges.append((f"#{a}", f"#{b}"))

hashtag_df = pd.DataFrame(hashtag_edges, columns=["node_u", "node_v"])
if not hashtag_df.empty:
    hashtag_df["weight"] = 1
    hashtag_df = hashtag_df.groupby(["node_u", "node_v"], as_index=False)["weight"].sum()

Gh = nx.Graph()
if not hashtag_df.empty:
    for _, r in hashtag_df.iterrows():
        Gh.add_edge(r["node_u"], r["node_v"], weight=int(r["weight"]))

# ╭──────────────────────────────────────────────────────────────╮
# │ 7️⃣ Compute centrality metrics                                │
# ╰──────────────────────────────────────────────────────────────╯
def centrality_table(G, kind: str):
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "degree", "betweenness", "pagerank", "type"])
    deg = dict(G.degree())
    bt = nx.betweenness_centrality(G, normalized=True, weight="weight") if G.number_of_edges() else {n: 0 for n in G.nodes()}
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {n: 1 / len(G) for n in G.nodes()}
    return pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [deg[n] for n in G.nodes()],
        "betweenness": [bt.get(n, 0) for n in G.nodes()],
        "pagerank": [pr.get(n, 0) for n in G.nodes()],
        "type": kind
    }).sort_values(["pagerank", "degree"], ascending=False)

mention_nodes = centrality_table(Gm, "mention")
hashtag_nodes = centrality_table(Gh, "hashtag")

# ╭──────────────────────────────────────────────────────────────╮
# │ 8️⃣ Save edges and nodes                                      │
# ╰──────────────────────────────────────────────────────────────╯
if not mention_df.empty:
    mention_df.to_csv(os.path.join(NETWORK_DIR, "mention_edges.csv"), index=False)
if not hashtag_df.empty:
    hashtag_df.to_csv(os.path.join(NETWORK_DIR, "hashtag_edges.csv"), index=False)

nodes_all = pd.concat([mention_nodes, hashtag_nodes], ignore_index=True)

def tag_node(node: str):
    s = str(node)
    if s.startswith("@"):
        return "user"
    if s.startswith("user:"):
        return "user_id"
    if s.startswith("#"):
        return "hashtag"
    return "other"

nodes_all["node_type"] = nodes_all["node"].apply(tag_node)
nodes_all.to_csv(os.path.join(NETWORK_DIR, "nodes_metrics.csv"), index=False)
print("🏷️ Saved nodes with auto-tagging → nodes_metrics.csv")

# ╭──────────────────────────────────────────────────────────────╮
# │ 9️⃣ Top accounts & hashtags                                   │
# ╰──────────────────────────────────────────────────────────────╯
top_accounts = nodes_all[nodes_all["node_type"].isin(["user", "user_id"])] \
    .nlargest(10, ["pagerank", "degree"])[["node", "degree", "pagerank", "betweenness", "node_type"]]
top_hashtags = nodes_all[nodes_all["node_type"] == "hashtag"] \
    .nlargest(10, ["pagerank", "degree"])[["node", "degree", "pagerank", "betweenness"]]

top_accounts.to_csv(os.path.join(NETWORK_DIR, "top_accounts.csv"), index=False)
top_hashtags.to_csv(os.path.join(NETWORK_DIR, "top_hashtags.csv"), index=False)
print("⭐ Exported top_accounts.csv and top_hashtags.csv")

# ╭──────────────────────────────────────────────────────────────╮
# │ 🔟 Optional interactive HTML (PyVis)                         │
# ╰──────────────────────────────────────────────────────────────╯
try:
    from pyvis.network import Network
    net = Network(height="750px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=150, spring_strength=0.02, damping=0.8)

    # Mention layer (blue)
    for n in Gm.nodes():
        net.add_node(n, label=n, color="#3b82f6")
    for u, v, d in Gm.edges(data=True):
        net.add_edge(u, v, value=d.get("weight", 1), color="#60a5fa")

    # Hashtag layer (green)
    for n in Gh.nodes():
        if n not in Gm:
            net.add_node(n, label=n, color="#10b981")
    for u, v, d in Gh.edges(data=True):
        net.add_edge(u, v, value=d.get("weight", 1), color="#34d399")

    html_path = os.path.join(NETWORK_DIR, "network_overview.html")
    net.write_html(html_path, open_browser=False)
    print(f"🌐 Interactive network saved → {html_path}")
except Exception as e:
    print(f"ℹ️ Skipped HTML visualization (pyvis not installed or other issue): {e}")
    print("   To enable, run: pip install pyvis")
