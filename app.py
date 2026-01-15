import feedparser
import html
from sklearn.cluster import AgglomerativeClustering
from flask import Flask, render_template, request, jsonify
import datetime
from datetime import timedelta
import ollama
import numpy as np
import ssl

# --- !! DANGER ZONE: UNSECURE MODE !! ---
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context
# --- END OF DANGER ZONE ---

# --- Configuration ---
CACHE_DURATION_SECONDS = 900
OLLAMA_EMBEDDING_MODEL = 'mxbai-embed-large:latest'
OLLAMA_GENERATION_MODEL = 'gemma3:4b' 

# --- Initialization ---
app = Flask(__name__)

# Global cache object
cache = {
    "data": None,
    "timestamp": None
}

print("--- System Startup ---")
print(f"Embedding Engine: Ollama ({OLLAMA_EMBEDDING_MODEL})")

def get_and_process_news():
    print("\n[DEBUG] ===== Starting get_and_process_news() =====")
    
    feeds_file = "feeds.txt"
    try:
        with open(feeds_file, "r") as f:
            feed_urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[DEBUG] CRITICAL: '{feeds_file}' not found.")
        return []

    if not feed_urls:
        return []

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    # --- 1. Define Cutoff Date (7 Days Ago) ---
    # We use UTC because feedparser usually returns time in UTC
    cutoff_date = datetime.datetime.utcnow() - timedelta(days=7)
    print(f"[DEBUG] Filtering articles older than: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    all_entries = []
    for url in feed_urls:
        print(f"[DEBUG] Fetching: {url}")
        try:
            feed = feedparser.parse(url, agent=USER_AGENT)
            if feed.entries:
                all_entries.extend(feed.entries)
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    if not all_entries:
        return []

    news_items = []
    seen_links = set()

    ignored_count = 0
    old_count = 0

    for entry in all_entries:
        # Deduplication check
        if entry.link in seen_links:
            continue
        seen_links.add(entry.link)

        # --- 2. Date Filtering Logic ---
        # feedparser provides 'published_parsed' as a time.struct_time (usually UTC)
        # If no date is present, we skip it to be safe.
        if not hasattr(entry, 'published_parsed') or not entry.published_parsed:
            ignored_count += 1
            continue

        try:
            # Convert struct_time to datetime object
            article_date = datetime.datetime(*entry.published_parsed[:6])
            
            # Check if article is too old
            if article_date < cutoff_date:
                old_count += 1
                continue
        except Exception:
            # If date parsing fails, skip
            ignored_count += 1
            continue

        # --- Content Extraction ---
        summary_text = ""
        if hasattr(entry, 'summary') and entry.summary:
            summary_text = entry.summary
        elif hasattr(entry, 'description') and entry.description:
            summary_text = entry.description

        summary = html.unescape(summary_text)
        title = html.unescape(entry.title)

        if not summary:
            continue

        image_url = None
        if 'media_content' in entry and entry.media_content:
            for media in entry.media_content:
                if 'medium' in media and media['medium'] == 'image' and 'url' in media:
                    image_url = media['url']
                    break
        
        # Format date for display
        formatted_date = article_date.strftime('%d. %B %Y, %H:%M Uhr')

        news_items.append({
            "title": title,
            "link": entry.link,
            "summary": summary,
            "image_url": image_url,
            "date": formatted_date
        })

    print(f"[DEBUG] Stats: Kept {len(news_items)} | Skipped {old_count} old articles | Skipped {ignored_count} undated/invalid.")
    
    if len(news_items) < 2:
        print("[DEBUG] Not enough recent items to cluster.")
        return []

    # --- Embedding (Title + Summary for better separation) ---
    texts_to_embed = [f"{item['title']}. {item['summary']}" for item in news_items]
    
    print(f"\n[DEBUG] Generating embeddings via Ollama ({OLLAMA_EMBEDDING_MODEL})...")
    
    embeddings = []
    try:
        for i, text in enumerate(texts_to_embed):
            clean_text = text[:4000] 
            response = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=clean_text)
            embeddings.append(response["embedding"])
        
        embeddings = np.array(embeddings)
    except Exception as e:
        print(f"[DEBUG] CRITICAL: Ollama embedding failed: {e}")
        return []

    print("[DEBUG] Clustering embeddings...")
    
    # --- Clustering Settings (Strict) ---
    clustering_model = AgglomerativeClustering(
        n_clusters=None, 
        metric='cosine', 
        linkage='complete', 
        distance_threshold=0.35 
    )
    cluster_assignments = clustering_model.fit_predict(embeddings)

    clustered_news = {}
    for index, cluster_id in enumerate(cluster_assignments):
        clustered_news.setdefault(cluster_id, []).append(news_items[index])

    sorted_clusters = sorted(clustered_news.values(), key=len, reverse=True)
    
    output_data = []
    for i, cluster_items in enumerate(sorted_clusters):
        output_data.append({
            "name": f"Topic {i+1}",
            "count": len(cluster_items),
            "articles": cluster_items
        })
    
    print(f"[DEBUG] Processing complete. Created {len(output_data)} clusters.")
    return output_data

@app.route('/')
def index():
    now = datetime.datetime.now()
    
    if cache["data"] and (now - cache["timestamp"]).total_seconds() < CACHE_DURATION_SECONDS:
        clusters_to_render = cache["data"]
    else:
        fresh_data = get_and_process_news()
        if fresh_data:
            cache["data"] = fresh_data
            cache["timestamp"] = now
        clusters_to_render = cache.get("data")
    
    return render_template('index.html', clusters=clusters_to_render, updated_time=cache.get("timestamp"))

@app.route('/summarize', methods=['POST'])
def summarize_cluster():
    data = request.get_json()
    if not data or 'articles' not in data:
        return jsonify({"error": "Invalid request"}), 400
    
    articles = data['articles']
    
    full_text = ""
    for article in articles:
        full_text += f"- {article['title']}: {article['summary'][:300]}\n"
    
    prompt = (
        f"Summarize these related news articles into one concise paragraph about the specific event:\n\n"
        f"{full_text}"
    )

    try:
        response = ollama.chat(
            model=OLLAMA_GENERATION_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return jsonify({"summary": response['message']['content']})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)