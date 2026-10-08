#!/bin/bash
# Start script for Frontend service on Render

echo "Starting Streamlit Frontend..."
echo "Port: $PORT"

# Start Streamlit with production settings
exec streamlit run ui/streamlit_app.py \
    --server.port ${PORT:-8501} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection true \
    --browser.gatherUsageStats false
