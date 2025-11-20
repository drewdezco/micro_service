#!/bin/bash
# Start both backend and frontend servers

echo "Starting both backend and frontend servers..."
echo "Backend will run on http://localhost:8000"
echo "Frontend will run on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Start backend in background
cd "$(dirname "$0")"
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend in background
API_URL=http://localhost:8000 uvicorn frontend.app:app --host 0.0.0.0 --port 3000 --reload &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

