# Smart-RSS-News-Viewer
Of course. Here is a complete and professional `README.md` file for your project repository. You can copy and paste this directly into a file named `README.md` in your project's root folder.

---

# AI-Powered News Summarizer

An intelligent, self-hosted web application that fetches articles from multiple RSS feeds, uses semantic clustering to group them by topic, and leverages a local Large Language Model (LLM) via Ollama to generate a concise, one-sentence summary for each topic.

The application features a minimalist, responsive, black-and-white UI designed for readability and focus, with AI-generated summaries loading asynchronously for a fast user experience.

---

## Key Features

- **Multi-Feed Aggregation**: Gathers news from an unlimited number of RSS feeds defined in a simple `feeds.txt` file.
- **Semantic Clustering**: Goes beyond keywords to understand the meaning of articles, grouping them into coherent topics using sentence-transformer embeddings.
- **AI-Powered Summarization**: Uses a local LLM (e.g., Google's Gemma) running via Ollama to generate a unique, single-sentence summary for each news cluster.
- **Asynchronous Loading**: The main interface loads almost instantly. AI summaries are then fetched and displayed one by one as they are generated, providing a highly responsive user experience.
- **Minimalist UI**: A clean, readable, monospace, black-and-white interface with a full-width layout and subtle outlines.
- **Interactive Experience**: AI summaries are highlighted with an animated rainbow effect. The underlying source articles for each summary can be expanded or collapsed with a single click.
- **Source Identification**: Favicons for each news source are displayed next to the AI summary, showing the origin of the articles within the cluster.
- **Self-Hosted & Private**: All processing, from embedding to summarization, happens on your local machine. No data is sent to external cloud services.

---

## Tech Stack

- **Backend**: Python 3, Flask
- **AI / Machine Learning**:
  - `sentence-transformers`: For generating semantic vector embeddings.
  - `scikit-learn`: For hierarchical agglomerative clustering.
  - **Ollama**: To serve the local Large Language Model.
- **Data Fetching**: `feedparser`
- **Frontend**: HTML5, CSS3, Vanilla JavaScript (for API calls)

---

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.8+** and `pip`.
2.  **Ollama**: You must have the [Ollama service](https://ollama.com/) installed and running on your machine.
3.  **An Ollama Model**: You need to have pulled the model specified in the application. For this project, we use `gemma3:4b`.
    ```bash
    ollama run gemma3:4b
    ```

---

## Installation & Setup

Follow these steps to get the application running.

**1. Clone the Repository**
```bash
git clone https://github.com/nicobo-crl/Smart-RSS-News-Viewer
cd Smart-RSS-News-Viewer
```

**2. Create the Project Structure**
Ensure your folder structure looks like this. The `index.html` file **must** be inside a `templates` sub-folder.
```
/
├── feeds.txt
├── app.py
└── templates/
    └── index.html
```

**3. Install Python Dependencies**
It is recommended to use a virtual environment.
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required libraries
pip install Flask feedparser sentence-transformers scikit-learn numpy requests
```

**4. Configure RSS Feeds**
Create a file named `feeds.txt` in the root of the project. Add the URLs of the RSS feeds you want to analyze, one URL per line.
```
# Example feeds.txt
https://www.tagesschau.de/index~rss2.xml
https://www.spiegel.de/schlagzeilen/tops/index.rss
https://www.heise.de/rss/heise-atom.xml
```

**5. Run the Application**

You need two terminal windows for this.

-   **In your first terminal**, make sure the Ollama service is running. If it's not already running as a background service, you can start it manually.
    ```bash
    ollama serve
    ```

-   **In your second terminal**, navigate to the project directory and run the Flask application.
    ```bash
    python app.py
    ```

---

## How to Use

1.  **Open Your Browser**: Navigate to `http://localhost:4000`.
2.  **View Summaries**: The page will load instantly and begin fetching AI summaries. The rainbow "Summary" text indicates AI-generated content.
3.  **Explore Sources**: Click the "Show Source Articles" button below any summary to see the list of articles that were used to generate it. Click again to collapse.
4.  **Change Feeds**: To add or remove news sources, simply edit the `feeds.txt` file and click the "Refresh & Reload Feeds" button in the browser.

---

## How It Works

1.  **Fetch & Cluster**: When you load the page, the Flask backend reads the URLs from `feeds.txt`, fetches all articles, converts their summaries into vector embeddings, and groups them into clusters based on semantic similarity.
2.  **Initial Render**: The server immediately sends the web page to your browser, showing all the clustered articles in a collapsed state but with placeholders for the AI summaries.
3.  **Async Summarization**: Once the page loads, JavaScript makes an API call to a local `/summarize` endpoint for each cluster.
4.  **LLM Inference**: The Flask server receives these requests, sends the combined text of the articles in a cluster to the Ollama API, and gets a one-sentence summary back from the Gemma model.
5.  **Dynamic Update**: The summary is sent back to the browser, where JavaScript replaces the "Loading..." placeholder with the final AI-generated text.

---

## Customization

-   **Change the AI Model**: Open `app.py` and change the `OLLAMA_MODEL` variable to any other model you have installed in Ollama (e.g., `llama3:8b`, `mistral`, etc.).
-   **Adjust Clustering**: In `app.py`, find the `AgglomerativeClustering` line. You can change the `distance_threshold` value (a smaller value like `0.5` will create more, smaller clusters; a larger value like `0.8` will create fewer, broader clusters).
-   **Modify the UI**: All styling is contained within the `<style>` block in `templates/index.html`. You can easily change fonts, colors, and layout here.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
