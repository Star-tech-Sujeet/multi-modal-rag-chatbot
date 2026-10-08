#!/bin/bash
# Build script for Render deployment

echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Creating necessary directories..."
mkdir -p data/uploads
mkdir -p data/chroma_db

echo "Build completed successfully!"
