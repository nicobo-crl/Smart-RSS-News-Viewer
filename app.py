import feedparser
import html
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from flask import Flask, render_template, request, jsonify
import datetime
from urllib.parse import urlparse
import requests

# --- Configuration ---
CACHE_DURATION_SECONDS = 900
FEEDS_FILENAME = "feeds.txt"
OLLAMA_HOST = "http://localhost:11434"
# CORRECTED: Using gemma:4b as requested. Make sure you have it with `ollama run gemma:4b`
OLLAMA_MODEL = "gemma3:4b"

# --- Initialization ---
app = Flask(__name__)
cache = {}

print("Loading high-performance sentence embedding model...")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print("Model loaded successfully.")

def summarize_cluster_with_ollama(cluster_articles):
    """Generates a short summary using Ollama."""
    print(f"  -> Summarizing a cluster of {len(cluster_articles)} articles with {OLLAMA_MODEL}...")
    full_text_content = ""
    for article in cluster_articles:
        full_text_content += f"Title: {article['title']}\nSummary: {article['summary']}\n\n"

    prompt = f"""
You are a news analyst AI. Synthesize the core event from the following articles into a single, brief sentence. Be direct and concise.

ARTICLES:
---
{full_text_content}
---
SUMMARY (1 SENTENCE):
"""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180
        )
        response.raise_for_status()
        summary = response.json().get("response", "Summary could not be generated.").strip()
        print("  -> Summary generated successfully.")
        return summary
    except requests.exceptions.RequestException as e:
        print(f"!!! ERROR connecting to Ollama: {e}")
        return "Error: Could not connect to the Ollama server. Is it running?"
    except Exception as e:
        print(f"!!! UNEXPECTED ERROR during summarization: {e}")
        return "Error: An unexpected error occurred during summarization."

def get_and_cluster_news(feed_urls):
    """Fetches, prepares, and clusters news, with favicon detection."""
    all_news_items = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            if feed.status != 200: continue
            for entry in feed.entries:
                summary = html.unescape(getattr(entry, 'summary', ''))
                if not summary: continue
                
                source_url = entry.get('link', url)
                domain = urlparse(source_url).netloc
                favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"

                formatted_date = None
                if 'published_parsed' in entry:
                    dt_obj = datetime.datetime(*entry.published_parsed[:6])
                    formatted_date = dt_obj.strftime('%d. %B %Y, %H:%M Uhr')
                all_news_items.append({
                    "title": entry.title, "link": entry.link, "summary": summary,
                    "date": formatted_date,
                    "source": feed.feed.get('title', url),
                    "favicon_url": favicon_url
                })
        except Exception as e:
            print(f"Error processing feed {url}: {e}")
    
    if len(all_news_items) < 2: return []

    print(f"Found {len(all_news_items)} articles. Clustering...")
    summaries = [item['summary'] for item in all_news_items]
    embeddings = model.encode(summaries, show_progress_bar=False)
    
    # CORRECTED: Using metric='cosine' as requested
    clustering_model = AgglomerativeClustering(
        n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.6
    )
    cluster_assignments = clustering_model.fit_predict(embeddings)

    clustered_news = {}
    for sentence_id, cluster_id in enumerate(cluster_assignments):
        if cluster_id not in clustered_news: clustered_news[cluster_id] = []
        clustered_news[cluster_id].append(all_news_items[sentence_id])

    sorted_clusters = sorted(clustered_news.values(), key=len, reverse=True)
    
    output_data = []
    for i, cluster_items in enumerate(sorted_clusters):
        unique_favicons = list(dict.fromkeys(item['favicon_url'] for item in cluster_items))
        
        output_data.append({
            "count": len(cluster_items),
            "articles": cluster_items,
            "favicons": unique_favicons,
        })

    print("Clustering complete. Ready to serve.")
    return output_data

@app.route('/')
def index():
    """Main route."""
    now = datetime.datetime.now()
    try:
        with open(FEEDS_FILENAME, 'r', encoding='utf-8') as f:
            feed_urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return "Error: feeds.txt not found. Please create it in the same directory as app.py.", 404
    
    if not feed_urls:
        return render_template('index.html', clusters=[])

    cache_key = "".join(sorted(feed_urls))
    if cache_key in cache:
        cached_item = cache[cache_key]
        if (now - cached_item["timestamp"]).total_seconds() < CACHE_DURATION_SECONDS:
            print("Serving clustered data from cache.")
            return render_template('index.html', clusters=cached_item["data"], updated_time=cached_item["timestamp"])

    print("Cache stale or empty. Processing clusters anew.")
    clustered_data = get_and_cluster_news(feed_urls)
    cache[cache_key] = {"data": clustered_data, "timestamp": now}
    return render_template('index.html', clusters=clustered_data, updated_time=now)

@app.route('/summarize', methods=['POST'])
def summarize():
    """API endpoint for summarization."""
    data = request.get_json()
    if not data or 'articles' not in data:
        return jsonify({"error": "Invalid request"}), 400
    
    articles = data['articles']
    summary = summarize_cluster_with_ollama(articles)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)