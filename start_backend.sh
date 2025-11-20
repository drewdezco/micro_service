#!/bin/bash
# Start the backend API server

echo "Starting backend API server on http://localhost:8000"
cd "$(dirname "$0")"
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

