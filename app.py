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

# --- !! DANGER ZONE: UNSECURE MODE !! ---
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context
# --- END OF DANGER ZONE ---

# --- Configuration ---
CACHE_DURATION_SECONDS = 3600  # 1 Hour
OLLAMA_EMBEDDING_MODEL = 'mxbai-embed-large:latest'
OLLAMA_GENERATION_MODEL = 'gemma3:4b' 

# --- Initialization ---
app = Flask(__name__)

# Global cache objects
cache = {
    "data": None,
    "timestamp": None
}
summary_cache = {} 

print("--- System Startup ---")
print(f"Embedding Engine: Ollama ({OLLAMA_EMBEDDING_MODEL})")
print(f"Summarization Engine: Ollama ({OLLAMA_GENERATION_MODEL})")

def get_favicon(url):
    """ Simple helper to get a favicon URL from a feed link """
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
    except:
        return ""

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
    cutoff_date = datetime.datetime.utcnow() - timedelta(days=7)
    
    all_entries = []
    # Map to store which feed domain an entry came from
    feed_domain_map = {} 

    for url in feed_urls:
        try:
            feed = feedparser.parse(url, agent=USER_AGENT)
            if feed.entries:
                # Capture domain for favicon
                domain_favicon = get_favicon(url)
                for entry in feed.entries:
                    feed_domain_map[entry.link] = domain_favicon
                all_entries.extend(feed.entries)
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    if not all_entries:
        return []

    news_items = []
    seen_links = set()

    for entry in all_entries:
        if entry.link in seen_links:
            continue
        seen_links.add(entry.link)

        if not hasattr(entry, 'published_parsed') or not entry.published_parsed:
            continue
        try:
            article_date = datetime.datetime(*entry.published_parsed[:6])
            if article_date < cutoff_date:
                continue
        except Exception:
            continue

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
        
        formatted_date = article_date.strftime('%d. %B %Y, %H:%M Uhr')
        
        # Get source name nicely
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
            "date": formatted_date,
            "source": source_name,
            "favicon": feed_domain_map.get(entry.link, "")
        })

    if len(news_items) < 2:
        return []

    # --- Embedding ---
    texts_to_embed = [f"{item['title']}. {item['summary']}" for item in news_items]
    
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

    # --- Clustering ---
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
        # Gather unique favicons for the cluster header
        unique_favicons = list(set([item['favicon'] for item in cluster_items if item['favicon']]))
        
        output_data.append({
            "name": f"Topic {i+1}",
            "count": len(cluster_items),
            "articles": cluster_items,
            "favicons": unique_favicons[:5] # Limit to 5 icons per header
        })
    
    return output_data

@app.route('/')
def index():
    now = datetime.datetime.now()
    
    # Cache Logic: If valid data exists, serve it immediately.
    if cache["data"] and cache["timestamp"] and (now - cache["timestamp"]).total_seconds() < CACHE_DURATION_SECONDS:
        clusters_to_render = cache["data"]
    else:
        fresh_data = get_and_process_news()
        if fresh_data:
            cache["data"] = fresh_data
            cache["timestamp"] = now
            summary_cache.clear() 
        clusters_to_render = cache.get("data")
    
    return render_template('index.html', clusters=clusters_to_render, updated_time=cache.get("timestamp"))

@app.route('/summarize', methods=['POST'])
def summarize_cluster():
    data = request.get_json()
    if not data or 'articles' not in data:
        return jsonify({"error": "Invalid request"}), 400
    
    articles = data['articles']
    article_links = sorted([a['link'] for a in articles])
    signature = hashlib.md5(json.dumps(article_links).encode('utf-8')).hexdigest()

    # Instant return if cached
    if signature in summary_cache:
        return jsonify({"summary": summary_cache[signature]})

    full_text = ""
    for article in articles:
        full_text += f"- {article['title']}: {article['summary'][:300]}\n"
    
    # --- UPDATED PROMPT ---
    prompt = (
        f"Summarize the following news articles into a single, concise paragraph. "
        f"Do not use introductory phrases like 'Here is a summary', 'The articles discuss', or 'In this text'. "
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