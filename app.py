import feedparser
import html
import os
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
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler

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
data_lock = threading.Lock()
global_store = {
    "clusters": None,
    "last_updated": None,
    "is_processing": False
}
summary_cache = {} 

print("--- System Startup ---")

def strip_html(html_content):
    """Remove HTML tags and return clean text"""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    # Get text and clean up whitespace
    text = soup.get_text(separator=' ', strip=True)
    # Normalize whitespace (multiple spaces/newlines to single space)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text

def get_favicon(url):
    """ Extracts domain from article URL to get the correct favicon """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if not domain:
            path_parts = parsed.path.split('/')
            if path_parts: domain = path_parts[0]
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
    except:
        return ""

def extract_image_url(entry):
    """ Aggressively tries to find an image in the RSS entry """
    if 'media_content' in entry:
        for media in entry.media_content:
            if (media.get('medium') == 'image' or media.get('type', '').startswith('image/')):
                if 'url' in media: return media['url']

    if 'media_thumbnail' in entry:
        thumbnails = entry.media_thumbnail
        if isinstance(thumbnails, list) and len(thumbnails) > 0:
            return thumbnails[0].get('url')

    if 'links' in entry:
        for link in entry.links:
            if link.get('rel') == 'enclosure' and link.get('type', '').startswith('image/'):
                return link['href']
    
    content = ""
    if 'summary' in entry: content = entry.summary
    elif 'description' in entry: content = entry.description
    
    if '<img' in content:
        try:
            start = content.find('src="') + 5
            if start > 4:
                end = content.find('"', start)
                if end > start: return content[start:end]
        except: pass

    return None

def process_news_workflow():
    with data_lock:
        if global_store["is_processing"]:
            print(f"[{datetime.datetime.now()}] Update already in progress. Skipping.")
            return None
        global_store["is_processing"] = True

    try:
        print(f"\n[{datetime.datetime.now()}] Starting Background Update...")

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

        for url in feed_urls:
            try:
                feed = feedparser.parse(url, agent=USER_AGENT)
                if feed.entries:
                    all_entries.extend(feed.entries)
            except Exception as e:
                print(f"Error fetching {url}: {e}")

        # 2. Filter & Format
        news_items = []
        seen_links = set()

        for entry in all_entries:
            if entry.link in seen_links: continue
            seen_links.add(entry.link)

            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    article_date = datetime.datetime(*entry.published_parsed[:6])
                    if article_date < cutoff_date: continue
                except: continue
            else:
                continue

            summary_text = ""
            if hasattr(entry, 'summary') and entry.summary: summary_text = entry.summary
            elif hasattr(entry, 'description') and entry.description: summary_text = entry.description

            # Strip HTML tags and unescape entities
            summary = strip_html(html.unescape(summary_text))
            title = strip_html(html.unescape(entry.title))

            if not summary: continue

            image_url = extract_image_url(entry)
            source_name = "Unknown"
            if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
                 source_name = entry.source.title
            elif 'link' in entry:
                source_name = urlparse(entry.link).netloc.replace('www.', '')

            news_items.append({
                "title": title,
                "link": entry.link,
                "summary": summary,
                "image_url": image_url,
                "date": article_date.strftime('%d. %B %Y, %H:%M Uhr'),
                "source": source_name,
                "favicon": get_favicon(entry.link)
            })

        if len(news_items) < 2:
            print("Not enough news items to cluster.")
            return []

        # 3. Embed (With Robust Error Handling)
        print(f"Embedding {len(news_items)} articles...")
        
        valid_items = []
        embeddings = []

        for item in news_items:
            text = f"{item['title']}. {item['summary']}"
            # CRITICAL FIX: Reduced from 4000 to 1500 to fit context window
            clean_text = text[:1500]

            try:
                response = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=clean_text)
                embeddings.append(response["embedding"])
                valid_items.append(item) # Only keep items that succeeded
            except Exception as e:
                print(f"Warning: Failed to embed '{item['title']}': {e}")
                continue

        if not embeddings:
            print("No embeddings generated. Aborting.")
            return None

        embeddings = np.array(embeddings)
        print(f"Successfully embedded {len(valid_items)} articles.")

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
            clustered_news.setdefault(cluster_id, []).append(valid_items[index])

        sorted_clusters = sorted(clustered_news.values(), key=len, reverse=True)

        output_data = []
        for i, cluster_items in enumerate(sorted_clusters):
            unique_favicons = list(set([item['favicon'] for item in cluster_items if item['favicon']]))
            output_data.append({
                "name": f"Topic {i+1}",
                "count": len(cluster_items),
                "articles": cluster_items,
                "favicons": unique_favicons[:6]
            })

        print(f"Update Complete. {len(output_data)} topics found.")

        with data_lock:
            global_store["clusters"] = output_data
            global_store["last_updated"] = datetime.datetime.now()
            summary_cache.clear()

        return output_data
    finally:
        with data_lock:
            global_store["is_processing"] = False

# --- APScheduler Setup ---
scheduler = BackgroundScheduler()

def scheduled_update():
    """Wrapper function for scheduled updates with error handling"""
    try:
        process_news_workflow()
    except Exception as e:
        print(f"Scheduled update failed: {e}")
        with data_lock:
            global_store["is_processing"] = False

# Add job to run every hour
scheduler.add_job(
    func=scheduled_update,
    trigger="interval",
    seconds=UPDATE_INTERVAL_SECONDS,
    id="news_update",
    replace_existing=True
)

# Start the scheduler and initial update
if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    # Start the scheduler
    scheduler.start()
    print(f"Scheduler started. Updates every {UPDATE_INTERVAL_SECONDS} seconds.")

    # Run initial update on startup in background thread
    initial_thread = threading.Thread(target=process_news_workflow, daemon=True)
    initial_thread.start()

# --- Routes ---

@app.route('/')
def index():
    with data_lock:
        data = global_store["clusters"]
        updated_at = global_store["last_updated"]

    if data is None:
        return render_template('loading.html')
    else:
        return render_template('index.html', clusters=data, updated_time=updated_at)

@app.route('/status')
def status():
    with data_lock:
        ready = global_store["clusters"] is not None
    return jsonify({"ready": ready})

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
