import feedparser
import html
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from flask import Flask, render_template
import datetime

# --- Configuration ---
CACHE_DURATION_SECONDS = 900  # 15 minutes

# --- Initialization ---
app = Flask(__name__)

# Global cache object
cache = {
    "data": None,
    "timestamp": None
}

# Load the model once on startup
print("Loading high-performance sentence embedding model...")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print("Model loaded successfully.")

def get_and_process_news():
    """
    Fetches, embeds, and clusters the news, with robust logic for finding images and dates.
    """
    print("Fetching and processing fresh news from the RSS feed...")
    feed_url = "https://rss.dw.com/rdf/rss-en-top"
    
    # 1. Fetching and Preprocessing
    feed = feedparser.parse(feed_url)
    if feed.status != 200:
        print(f"Error fetching feed: Status {feed.status}")
        return []

    news_items = []
    for entry in feed.entries:
        summary = html.unescape(entry.summary)
        if not summary:
            continue

        # --- UPDATED & MORE ROBUST IMAGE EXTRACTION ---
        image_url = None
        
        # Method 1: Check 'media_content' (very common for images)
        if 'media_content' in entry and entry.media_content:
            # entry.media_content is a list, find the one that is an image
            for media in entry.media_content:
                if 'medium' in media and media['medium'] == 'image' and 'url' in media:
                    image_url = media['url']
                    break
        
        # Method 2: If no image yet, check 'enclosures' (for podcasts, videos, or images)
        if not image_url and 'enclosures' in entry and entry.enclosures:
            for enc in entry.enclosures:
                if 'type' in enc and enc.type.startswith('image/'):
                    image_url = enc.href
                    break
        
        # Method 3: If no image yet, check 'links' for an image relationship
        if not image_url and 'links' in entry and entry.links:
            for link in entry.links:
                if 'type' in link and link.type.startswith('image/'):
                    image_url = link.href
                    break
        
        # --- End of Updated Image Extraction ---

        # Extract and Format Publication Date
        formatted_date = None
        if 'published_parsed' in entry:
            dt_obj = datetime.datetime(*entry.published_parsed[:6])
            formatted_date = dt_obj.strftime('%d. %B %Y, %H:%M Uhr')

        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "summary": summary,
            "image_url": image_url,
            "date": formatted_date
        })

    if len(news_items) < 2:
        return []

    summaries = [item['summary'] for item in news_items]

    # 2. Embedding and 3. Clustering
    print("Calculating embeddings and clustering news...")
    embeddings = model.encode(summaries, show_progress_bar=False)
    
    distance_threshold = 0.6
    clustering_model = AgglomerativeClustering(
        n_clusters=None, metric='cosine', linkage='average', distance_threshold=distance_threshold
    )
    cluster_assignments = clustering_model.fit_predict(embeddings)

    # 4. Grouping and Formatting
    clustered_news = {}
    for sentence_id, cluster_id in enumerate(cluster_assignments):
        if cluster_id not in clustered_news:
            clustered_news[cluster_id] = []
        clustered_news[cluster_id].append(news_items[sentence_id])

    sorted_clusters = sorted(clustered_news.values(), key=len, reverse=True)

    output_data = []
    for i, cluster_items in enumerate(sorted_clusters):
        cluster_name = f"Cluster {i+1}"
        if len(cluster_items) == 1:
            cluster_name = f"Unique Topic {i+1}"
        
        output_data.append({
            "name": cluster_name,
            "count": len(cluster_items),
            "articles": cluster_items
        })

    print("News processing complete.")
    return output_data

@app.route('/')
def index():
    """ Handles web requests and serves the main page with caching. """
    now = datetime.datetime.now()
    
    if cache["data"] and cache["timestamp"]:
        age = (now - cache["timestamp"]).total_seconds()
        if age < CACHE_DURATION_SECONDS:
            print("Serving news from cache.")
            return render_template('index.html', clusters=cache["data"], updated_time=cache["timestamp"])

    print("Cache is stale or empty. Fetching new data.")
    fresh_data = get_and_process_news()
    
    cache["data"] = fresh_data
    cache["timestamp"] = now

    return render_template('index.html', clusters=fresh_data, updated_time=now)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)