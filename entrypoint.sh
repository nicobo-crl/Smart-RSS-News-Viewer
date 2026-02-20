#!/bin/sh
set -e

# Wait for Ollama to be ready
echo "Waiting for Ollama..."
until curl -s http://ollama:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done

# Pull required models if not present
echo "Checking/Installing models..."
curl -s http://ollama:11434/api/pull -d '{"name": "mxbai-embed-large"}' > /dev/null
curl -s http://ollama:11434/api/pull -d '{"name": "gemma3:4b"}' > /dev/null

echo "Models ready. Starting app..."
exec python app.py