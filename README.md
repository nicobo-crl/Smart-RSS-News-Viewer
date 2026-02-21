# AI-Powered News Summarizer


An intelligent, self-hosted web application that fetches articles from multiple RSS feeds, uses semantic clustering to group them by topic, and leverages a local Large Language Model (LLM) via Ollama to generate a concise summary for each topic.
---

## Demo

Try it out here: **[https://news.boehringernico.de](http://news.boehringernico.de)**
*(Note: You may need to wait a minute for the server to prepare the news if it is waking up.)*

---

## Key Features

-   **Multi-Feed Aggregation**: Gathers news from an unlimited number of RSS feeds defined in a simple `feeds.txt` file.
-   **AI-Powered Summarization**: Uses a local LLM (Gemma 3) via Ollama to generate a unique, single-sentence summary for each cluster.
-   **Zero-Wait Caching**: A background worker fetches and processes news every hour. Users are served the latest cached version instantly.
-   **Robust Image Extraction**: Aggressively finds images from various RSS metadata standards (Media content, Enclosures, Thumbnails).
-   **Self-Hosted & Private**: All processing happens on your local machine or server. No data is sent to external cloud services.

---

## Installation & Setup (Docker)

This is the recommended way to run the application.

**1. Clone the Repository**
```bash
git clone https://github.com/nicobo-crl/Smart-RSS-News-Viewer
cd Smart-RSS-News-Viewer
```

**2. Start the Services**
Run the application in detached mode. This will start the Flask app and the Ollama service.
```bash
docker-compose up -d
```

**3. Download AI Models**
Because the AI models are large, they are not included in the Docker image. You must pull them into the running Ollama container manually.

Run these commands in your terminal **while the containers are running**:

1.  **Pull the Embedding Model** (Used for clustering topics):
    ```bash
    docker exec -it ollama-service ollama pull mxbai-embed-large:latest
    ```
    
2.  **Pull the Generation Model** (Used for writing summaries):
    ```bash
    docker exec -it ollama-service ollama pull gemma3:4b
    ```

**4. Access the App**
*   Open your browser to: `http://localhost:4000`
*   **First Run:** You will see a "Initializing System" loading screen while the app processes the first batch of news.
*   **Subsequent Runs:** The page will load instantly.

---

## Configuration

**Manage Feeds**
To change the news sources, simply edit the `feeds.txt` file in the root directory. Add one RSS URL per line.

```text
# Example feeds.txt
http://feeds.bbci.co.uk/news/world/rss.xml
https://www.theguardian.com/world/rss
https://rss.nytimes.com/services/xml/rss/nyt/World.xml
https://www.cnbc.com/id/100727362/device/rss/rss.html
```
*Note: After changing feeds, you can restart the container (`docker-compose restart`) to force an immediate update, or wait for the hourly background refresh.*

---

## Manual Setup (Without Docker)

If you prefer to run this directly on your host machine (Mac/Windows/Linux):

1.  **Install Ollama**: Download from [ollama.com](https://ollama.com).
2.  **Pull Models**:
    ```bash
    ollama pull mxbai-embed-large:latest
    ollama pull gemma3:4b
    ```
3.  **Install Python Deps**:
    ```bash
    pip install Flask feedparser sentence-transformers scikit-learn numpy requests
    ```
4.  **Run**:
    ```bash
    python app.py
    ```

---

## Monitoring & Troubleshooting

### Check Background Updates
The application logs its progress to the console. You can monitor the background worker by checking the Docker logs:

```bash
docker-compose logs -f news-app
```

You should see messages like:
- `[timestamp] Starting Background Update...`
- `Embedding X articles...`
- `Update Complete. X topics found.`

### Manual Refresh
If you want to force an update immediately:
1.  **Restart the Container**: `docker-compose restart news-app`
2.  **Wait for the Page**: The first user to visit after a restart will see the "Initializing System" screen while the first batch is processed.

---

## License

This project is licensed under the MIT License.



