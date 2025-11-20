# 🧠 Toxicity Classifier

A microservice for classifying text as toxic or non-toxic, with separate frontend and backend services.

## 📁 Project Structure

```
micro_service/
├── src/              # Backend API
│   └── app.py       # FastAPI backend application
├── frontend/         # Frontend UI
│   ├── app.py       # FastAPI frontend server
│   └── index.html   # Frontend HTML/CSS/JS
├── models/          # ML models
├── data/            # Training data (CSV files)
├── tests/           # Test files
└── docs/            # Documentation
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Create and Activate Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model (First Time Only)

If you haven't trained the model yet, or want to retrain on your data:

```bash
python train_model.py
```

This will train the model using data from `data/train.csv` and save it to `models/toxicity_model.joblib`.

## 🚀 Running the Application

### Option 1: Start Both Servers Together (Recommended)

```bash
./start_all.sh
```

This will start:
- **Backend API** on `http://localhost:8000`
- **Frontend UI** on `http://localhost:3000`

### Option 2: Start Servers Separately

**Terminal 1 - Backend:**
```bash
./start_backend.sh
# or manually:
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
./start_frontend.sh
# or manually:
API_URL=http://localhost:8000 uvicorn frontend.app:app --host 0.0.0.0 --port 3000 --reload
```

## 🌐 Accessing the Application

- **Frontend UI**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs
- **Backend Health Check**: http://localhost:8000/healthz

## 🔧 Configuration

### Backend Port
Default: `8000`
Change by modifying `start_backend.sh` or the uvicorn command.

### Frontend Port
Default: `3000`
Change by modifying `start_frontend.sh` or the uvicorn command.

### API URL (for frontend)
The frontend needs to know where the backend API is running. Set the `API_URL` environment variable:

```bash
API_URL=http://localhost:8000 uvicorn frontend.app:app --port 3000
```

If not set, it defaults to `http://localhost:8000`.

## 📝 Quick Setup Summary

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (if needed)
python train_model.py

# 4. Start both servers
./start_all.sh
```

Then open your browser to `http://localhost:3000` to use the application!

## 📡 API Endpoints

### POST `/predict`
Classify text for toxicity.

**Query Parameters:**
- `include_rationale` (bool): Include rationale spans (default: `true`)
- `redact_flagged` (bool): Return redacted text (default: `false`)
- `threshold` (float): Classification threshold 0.0-1.0 (default: `0.5`)

**Request Body:**
```json
{
  "text": "Your text here"
}
```

**Response:**
```json
{
  "label": "toxic",
  "confidence": 0.95,
  "scores": {
    "non_toxic": 0.05,
    "toxic": 0.95
  },
  "rationale": [
    {
      "span": "idiot",
      "start": 8,
      "end": 13,
      "weight": 1.0
    }
  ],
  "redacted_text": "You are an [REDACTED]!",
  "meta": {
    "latency_ms": 12.34,
    "threshold_used": 0.5
  }
}
```

### GET `/healthz`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_version": "toy-0.1"
}
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

## 📝 Development Notes

- The frontend and backend are completely separated, allowing you to:
  - Deploy them independently
  - Support multiple frontends (web, mobile, etc.)
  - Scale them separately
  - Use different technologies for each

- The frontend communicates with the backend via HTTP requests to the `/predict` endpoint.

- CORS is configured to allow requests from `localhost:3000`, `127.0.0.1:3000`, and `0.0.0.0:3000`.

- The rationale feature uses the model's feature importance to identify toxic words, with filtering to exclude common stop words.

## 🔐 Future Enhancements

- Add authentication/authorization
- Add rate limiting
- Add logging and monitoring
- Add Docker containers for easy deployment
- Improve rationale detection with better NLP techniques

