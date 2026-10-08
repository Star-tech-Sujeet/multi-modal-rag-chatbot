#!/bin/bash
# Start script for Backend service on Render

echo "Starting Multi-Modal RAG Backend..."
echo "Port: $PORT"

# Create data directories if they don't exist
mkdir -p data/uploads
mkdir -p data/chroma_db

# Start the uvicorn server
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
