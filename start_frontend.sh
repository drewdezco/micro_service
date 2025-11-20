#!/bin/bash
# Start the frontend UI server

echo "Starting frontend UI server on http://localhost:3000"
echo "Backend API URL: ${API_URL:-http://localhost:8000}"
cd "$(dirname "$0")"
API_URL=${API_URL:-http://localhost:8000} uvicorn frontend.app:app --host 0.0.0.0 --port 3000 --reload

