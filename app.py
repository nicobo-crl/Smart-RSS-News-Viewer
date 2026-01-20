import feedparser
import html
from sklearn.cluster import AgglomerativeClustering
from flask import Flask, render_template, request, jsonify
import datetime
from datetime import timedelta
import ollama
import numpy as np
import ssl
import hashlib
import json
import threading
import time
import copy

# --- !! DANGER ZONE: UNSECURE MODE !! ---
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context
# --- END OF DANGER ZONE ---

# --- Configuration ---
UPDATE_INTERVAL_SECONDS = 3600  # 1 Hour
OLLAMA_EMBEDDING_MODEL = 'mxbai-embed-large:latest'
OLLAMA_GENERATION_MODEL = 'gemma3:4b' 

# --- Initialization ---
app = Flask(__name__)

# Global Store
# We use a lock to ensure we don't read data while it's being written
data_lock = threading.Lock()
global_store = {
    "clusters": None,      # Stores the processed clusters
    "last_updated": None,  # Timestamp
    "is_processing": False # Flag to show status on loading screen
}
summary_cache = {} 

print("--- System Startup ---")

def get_favicon(url):
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
    except:
        return ""

def process_news_workflow():
    """ 
    This function performs the heavy lifting: 
    Fetching -> Embedding -> Clustering 
    """
    print(f"\n[{datetime.datetime.now()}] Starting Background Update...")
    
    with data_lock:
        global_store["is_processing"] = True

    feeds_file = "feeds.txt"
    try:
        with open(feeds_file, "r") as f:
            feed_urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"CRITICAL: '{feeds_file}' not found.")
        return None

    if not feed_urls:
        return None

    # 1. Fetch
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    cutoff_date = datetime.datetime.utcnow() - timedelta(days=7)
    all_entries = []
    feed_domain_map = {} 

    for url in feed_urls:
        try:
            feed = feedparser.parse(url, agent=USER_AGENT)
            if feed.entries:
                domain_favicon = get_favicon(url)
                for entry in feed.entries:
                    feed_domain_map[entry.link] = domain_favicon
                all_entries.extend(feed.entries)
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    # 2. Filter & Format
    news_items = []
    seen_links = set()

    for entry in all_entries:
        if entry.link in seen_links: continue
        seen_links.add(entry.link)

        # Date Check
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                article_date = datetime.datetime(*entry.published_parsed[:6])
                if article_date < cutoff_date: continue
            except: continue
        else:
            continue # Skip if no date

        # Text Extraction
        summary_text = ""
        if hasattr(entry, 'summary') and entry.summary: summary_text = entry.summary
        elif hasattr(entry, 'description') and entry.description: summary_text = entry.description
        
        summary = html.unescape(summary_text)
        title = html.unescape(entry.title)
        if not summary: continue

        image_url = None
        if 'media_content' in entry and entry.media_content:
            for media in entry.media_content:
                if 'medium' in media and media['medium'] == 'image' and 'url' in media:
                    image_url = media['url']
                    break
        
        # Source Name
        source_name = "Unknown"
        if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
             source_name = entry.source.title
        elif 'link' in entry:
            from urllib.parse import urlparse
            source_name = urlparse(entry.link).netloc.replace('www.', '')

        news_items.append({
            "title": title,
            "link": entry.link,
            "summary": summary,
            "image_url": image_url,
            "date": article_date.strftime('%d. %B %Y, %H:%M Uhr'),
            "source": source_name,
            "favicon": feed_domain_map.get(entry.link, "")
        })

    if len(news_items) < 2:
        print("Not enough news items to cluster.")
        return []

    # 3. Embed
    print(f"Embedding {len(news_items)} articles...")
    texts_to_embed = [f"{item['title']}. {item['summary']}" for item in news_items]
    embeddings = []
    try:
        for text in texts_to_embed:
            # Truncate to avoid context limit errors
            response = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text[:4000])
            embeddings.append(response["embedding"])
        embeddings = np.array(embeddings)
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

    # 4. Cluster
    print("Clustering...")
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
        unique_favicons = list(set([item['favicon'] for item in cluster_items if item['favicon']]))
        output_data.append({
            "name": f"Topic {i+1}",
            "count": len(cluster_items),
            "articles": cluster_items,
            "favicons": unique_favicons[:5]
        })
    
    print(f"Update Complete. {len(output_data)} topics found.")
    return output_data

def background_scheduler():
    """ Runs in a separate thread. Updates data, then sleeps. """
    while True:
        try:
            new_data = process_news_workflow()
            
            if new_data is not None:
                with data_lock:
                    global_store["clusters"] = new_data
                    global_store["last_updated"] = datetime.datetime.now()
                    # We clear summary cache on new data so we don't show old summaries for new topics
                    summary_cache.clear()
                    global_store["is_processing"] = False
            else:
                # If update failed, we keep old data but turn off processing flag
                with data_lock:
                    global_store["is_processing"] = False
                    
        except Exception as e:
            print(f"Background worker crashed: {e}")
            with data_lock:
                global_store["is_processing"] = False
        
        # Sleep for 1 hour before next update
        time.sleep(UPDATE_INTERVAL_SECONDS)

# Start the background thread immediately
t = threading.Thread(target=background_scheduler, daemon=True)
t.start()

# --- Routes ---

@app.route('/')
def index():
    # Check if we have data ready
    with data_lock:
        data = global_store["clusters"]
        updated_at = global_store["last_updated"]

    if data is None:
        # Initial Cold Start: Show Loading Screen
        return render_template('loading.html')
    else:
        # Hot Cache: Show News Immediately
        return render_template('index.html', clusters=data, updated_time=updated_at)

@app.route('/status')
def status():
    """ Endpoint for the loading screen to poll """
    with data_lock:
        ready = global_store["clusters"] is not None
    return jsonify({"ready": ready})

@app.route('/force_refresh')
def force_refresh():
    """ Optional: Trigger manual refresh (resets thread loop essentially) """
    # In a simple thread model, forcing is tricky without complex logic. 
    # For now, we just tell the user to wait for the next cycle or restart container.
    return "Background refresh is automatic. Restart container to force immediate update."

@app.route('/summarize', methods=['POST'])
def summarize_cluster():
    data = request.get_json()
    if not data or 'articles' not in data:
        return jsonify({"error": "Invalid request"}), 400
    
    articles = data['articles']
    article_links = sorted([a['link'] for a in articles])
    signature = hashlib.md5(json.dumps(article_links).encode('utf-8')).hexdigest()

    if signature in summary_cache:
        return jsonify({"summary": summary_cache[signature]})

    full_text = ""
    for article in articles:
        full_text += f"- {article['title']}: {article['summary'][:300]}\n"
    
    prompt = (
        f"Summarize the following news articles into a single, concise paragraph. "
        f"Do not use introductory phrases like 'Here is a summary'. "
        f"Start directly with the main actor, event, or fact.\n\n"
        f"{full_text}"
    )

    try:
        response = ollama.chat(
            model=OLLAMA_GENERATION_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        summary = response['message']['content']
        summary_cache[signature] = summary
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)