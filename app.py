import feedparser
import html
from sklearn.cluster import AgglomerativeClustering
# NEW: Import request and jsonify from Flask
from flask import Flask, render_template, request, jsonify
import datetime
import ollama
import numpy as np

# --- Configuration ---
CACHE_DURATION_SECONDS = 900  # 15 minutes
OLLAMA_EMBEDDING_MODEL = 'mxbai-embed-large:latest'
# NEW: Define a model for generating summaries
OLLAMA_GENERATION_MODEL = 'phi4:latest' 

# --- Initialization ---
app = Flask(__name__)

# Global cache object
cache = {
    "data": None,
    "timestamp": None
}

print("Using Ollama for sentence embeddings.")
print(f"Embedding model: {OLLAMA_EMBEDDING_MODEL}")
print(f"Generation model: {OLLAMA_GENERATION_MODEL}")


def get_and_process_news():
    """
    Fetches, embeds using Ollama, and clusters the news, with robust logic and heavy debugging.
    """
    print("\n[DEBUG] ===== Starting get_and_process_news() =====")
    feed_url = "https://rss.dw.com/rdf/rss-en-top"
    
    # 1. Fetching and Preprocessing
    print(f"[DEBUG] Fetching RSS feed from: {feed_url}")
    feed = feedparser.parse(feed_url)
    
    if feed.status != 200:
        print(f"[DEBUG] CRITICAL: Error fetching feed! Status code: {feed.status}")
        return []

    print(f"[DEBUG] Feed fetch successful. Found {len(feed.entries)} entries.")
    if not feed.entries:
        print("[DEBUG] CRITICAL: Feed is empty. Cannot process any news.")
        return []

    news_items = []
    for i, entry in enumerate(feed.entries):
        print(f"\n[DEBUG] --- Processing Entry {i+1}/{len(feed.entries)} ---")
        print(f"[DEBUG] Title: {entry.get('title', 'N/A')}")

        summary_text = ""
        has_summary = hasattr(entry, 'summary') and entry.summary
        has_description = hasattr(entry, 'description') and entry.description

        print(f"[DEBUG] 'summary' field exists and is not empty: {bool(has_summary)}")
        print(f"[DEBUG] 'description' field exists and is not empty: {bool(has_description)}")
        
        if has_summary:
            summary_text = entry.summary
            print("[DEBUG] Using 'summary' field.")
        elif has_description:
            summary_text = entry.description
            print("[DEBUG] Falling back to 'description' field.")
        else:
            print("[DEBUG] No 'summary' or 'description' field found for this entry.")

        summary = html.unescape(summary_text)

        if not summary:
            print(f"[DEBUG] Skipping entry because its summary/description is empty.")
            continue
        
        print(f"[DEBUG] Final summary text (first 80 chars): {summary[:80]}...")

        image_url = None
        if 'media_content' in entry and entry.media_content:
            for media in entry.media_content:
                if 'medium' in media and media['medium'] == 'image' and 'url' in media:
                    image_url = media['url']
                    break
        print(f"[DEBUG] Image URL found: {image_url}")

        formatted_date = None
        if 'published_parsed' in entry:
            dt_obj = datetime.datetime(*entry.published_parsed[:6])
            formatted_date = dt_obj.strftime('%d. %B %Y, %H:%M Uhr')
        print(f"[DEBUG] Date found: {formatted_date}")

        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "summary": summary,
            "image_url": image_url,
            "date": formatted_date
        })
        print(f"[DEBUG] >>> Successfully added entry to news_items list.")

    print(f"\n[DEBUG] Total news items successfully processed: {len(news_items)}")
    
    if len(news_items) < 2:
        print("[DEBUG] CRITICAL: Not enough news items with summaries to perform clustering. Aborting.")
        return []

    summaries = [item['summary'] for item in news_items]

    print(f"\n[DEBUG] Calculating embeddings with Ollama for {len(summaries)} articles...")
    try:
        embeddings = [ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=s)["embedding"] for s in summaries]
        embeddings = np.array(embeddings)
    except Exception as e:
        print(f"[DEBUG] CRITICAL: An error occurred during embedding: {e}")
        return []

    print("[DEBUG] Clustering news...")
    distance_threshold = 0.6
    clustering_model = AgglomerativeClustering(
        n_clusters=None, metric='cosine', linkage='average', distance_threshold=distance_threshold
    )
    cluster_assignments = clustering_model.fit_predict(embeddings)

    clustered_news = {}
    for sentence_id, cluster_id in enumerate(cluster_assignments):
        clustered_news.setdefault(cluster_id, []).append(news_items[sentence_id])

    sorted_clusters = sorted(clustered_news.values(), key=len, reverse=True)
    
    output_data = []
    for i, cluster_items in enumerate(sorted_clusters):
        output_data.append({
            "name": f"Cluster {i+1}" if len(cluster_items) > 1 else f"Unique Topic {i+1}",
            "count": len(cluster_items),
            "articles": cluster_items
        })
    
    print(f"[DEBUG] News processing complete. Found {len(output_data)} clusters.")
    print("[DEBUG] ===== Finished get_and_process_news() =====")
    return output_data

@app.route('/')
def index():
    """ Handles web requests and serves the main page with caching. """
    now = datetime.datetime.now()
    
    if cache["data"] and (now - cache["timestamp"]).total_seconds() < CACHE_DURATION_SECONDS:
        print("Serving news from cache.")
        clusters_to_render = cache["data"]
    else:
        print("Cache is stale or empty. Fetching new data.")
        fresh_data = get_and_process_news()
        if fresh_data:
            cache["data"] = fresh_data
            cache["timestamp"] = now
        clusters_to_render = cache.get("data")

    print("\n[DEBUG] --- Preparing to render template ---")
    if clusters_to_render:
        print(f"[DEBUG] Data is available. Passing {len(clusters_to_render)} clusters to the template.")
    else:
        print("[DEBUG] CRITICAL: No cluster data is available to pass to the template. Page will likely show an error or be empty.")
    
    return render_template('index.html', clusters=clusters_to_render, updated_time=cache.get("timestamp"))


# NEW: Add the missing /summarize route
@app.route('/summarize', methods=['POST'])
def summarize_cluster():
    """
    Receives articles for a cluster, generates a summary using Ollama, and returns it.
    """
    print("\n[DEBUG] ===== Received request on /summarize endpoint =====")
    
    # 1. Get the data from the request
    data = request.get_json()
    if not data or 'articles' not in data:
        print("[DEBUG] CRITICAL: Invalid data received. 'articles' key is missing.")
        return jsonify({"error": "Invalid request"}), 400
    
    articles = data['articles']
    print(f"[DEBUG] Received {len(articles)} articles to summarize.")

    # 2. Combine the titles and summaries into a single text block
    full_text = ""
    for article in articles:
        full_text += article['title'] + ". " + article['summary'] + "\n\n"
    
    print(f"[DEBUG] Combined text for summarization (first 150 chars): {full_text[:150]}...")

    # 3. Create a prompt for the Ollama model
    prompt = f"Please summarize the following news articles into a concise paragraph. Capture the main theme and key points:\n\n{full_text}"

    try:
        print(f"[DEBUG] Sending request to Ollama generation model: {OLLAMA_GENERATION_MODEL}")
        response = ollama.chat(
            model=OLLAMA_GENERATION_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        summary = response['message']['content']
        print(f"[DEBUG] Received summary from Ollama: {summary[:100]}...")
        
        # 4. Return the summary in a JSON response
        print("[DEBUG] ===== Successfully generated summary. Sending response. =====")
        return jsonify({"summary": summary})

    except Exception as e:
        print(f"[DEBUG] CRITICAL: An error occurred during summarization with Ollama: {e}")
        print("[DEBUG] ===== Failed to generate summary. Sending error response. =====")
        return jsonify({"error": "Failed to generate summary"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
