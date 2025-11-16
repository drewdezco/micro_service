# 🧠 Toxicity Classifier API

Welcome to the **Toxicity Classifier API** - a machine learning microservice that analyzes text for toxic content and provides detailed insights about potentially harmful language.

## 📋 Overview

This is a FastAPI-based microservice that classifies text as toxic or non-toxic using a machine learning model. It provides confidence scores, identifies problematic words, and can optionally redact toxic content.

### Key Features

- **🎯 Binary Classification**: Determines if text is toxic or non-toxic
- **📊 Confidence Scores**: Returns probability scores for both classes
- **🔍 Rationale Detection**: Identifies specific words/phrases that triggered the classification
- **🔒 Content Redaction**: Optionally masks toxic words with `[REDACTED]` or asterisks
- **⚡ Fast Response**: Low-latency predictions with performance metrics
- **📝 Interactive Documentation**: Auto-generated API docs with Swagger UI
- **✅ Health Monitoring**: Built-in health check endpoint

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/drewdezco/micro_service.git
   cd micro_service
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Start the server**
   ```bash
   cd src
   uvicorn app:app --reload
   ```

5. **Access the API**
   - Interactive Docs: http://127.0.0.1:8000/docs
   - Alternative Docs: http://127.0.0.1:8000/redoc
   - Health Check: http://127.0.0.1:8000/healthz

## 📚 API Endpoints

### `GET /healthz`

Health check endpoint to verify the service is running.

**Response:**
```json
{
  "status": "ok",
  "model_version": "toy-0.1"
}
```

### `POST /predict`

Classify text for toxicity with optional rationale and redaction.

**Query Parameters:**
- `include_rationale` (bool, default: true) - Include toxic word spans
- `redact_flagged` (bool, default: false) - Return redacted version of text
- `threshold` (float, default: 0.5) - Classification threshold (0.0-1.0)

**Request Body:**
```json
{
  "text": "Your text to analyze here"
}
```

**Response:**
```json
{
  "label": "toxic",
  "confidence": 0.94,
  "scores": {
    "non_toxic": 0.06,
    "toxic": 0.94
  },
  "rationale": [
    {
      "span": "idiot",
      "start": 11,
      "end": 16,
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

## 💡 Usage Examples

### Using cURL

```bash
# Basic prediction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Thank you for your help"}'

# With rationale and redaction
curl -X POST "http://127.0.0.1:8000/predict?include_rationale=true&redact_flagged=true" \
  -H "Content-Type: application/json" \
  -d '{"text":"You are an idiot!"}'

# Custom threshold
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.7" \
  -H "Content-Type: application/json" \
  -d '{"text":"This is borderline content"}'
```

### Using Python

```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {"text": "You are an idiot!"}
params = {"include_rationale": True, "redact_flagged": True}

response = requests.post(url, json=payload, params=params)
result = response.json()

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Redacted: {result['redacted_text']}")
```

### Using JavaScript

```javascript
const response = await fetch('http://127.0.0.1:8000/predict?include_rationale=true', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Your text here' })
});

const result = await response.json();
console.log(result);
```

## 🏗️ Project Structure

```
micro_service/
├── docs/                      # Documentation files
│   ├── README.md             # Detailed project guide
│   └── PROJECT_DESCRIPTION.md
├── src/                       # Source code
│   └── app.py                # FastAPI application
├── models/                    # Trained models
│   └── toxicity_model.joblib # Serialized ML model
├── tests/                     # Test suite
│   └── test_app.py           # API tests
├── train_model.py            # Model training script
└── requirements.txt          # Python dependencies
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

All tests should pass:
- ✅ Health check endpoint
- ✅ Basic prediction
- ✅ Toxic content detection
- ✅ Rationale extraction
- ✅ Text redaction
- ✅ Input validation

## 🔧 Technology Stack

- **FastAPI** - Modern, fast web framework for building APIs
- **scikit-learn** - Machine learning library for model training
- **TF-IDF Vectorizer** - Text feature extraction
- **Logistic Regression** - Classification algorithm
- **Pydantic** - Data validation using Python type annotations
- **Uvicorn** - Lightning-fast ASGI server
- **Pytest** - Testing framework

## 📊 Model Details

The current model is a simple baseline classifier:
- **Algorithm**: Logistic Regression
- **Features**: TF-IDF (1-2 grams)
- **Training Data**: 10 sample sentences (5 toxic, 5 non-toxic)
- **Purpose**: Demonstration and portfolio project

### Model Performance

This is a **toy model** for demonstration purposes. For production use, consider:
- Training on larger datasets (e.g., Jigsaw Toxic Comments)
- Using transformer-based models (BERT, RoBERTa)
- Implementing proper evaluation metrics
- Adding bias detection and mitigation

## 🎯 Use Cases

- **Content Moderation**: Filter user-generated content in forums, comments, or chat
- **Social Media Monitoring**: Detect harmful language in posts and messages
- **Customer Support**: Flag aggressive or abusive customer interactions
- **Educational Tools**: Help users understand and improve their communication
- **Research**: Analyze toxicity patterns in text datasets

## 🚦 API Response Codes

- `200 OK` - Successful prediction
- `422 Unprocessable Entity` - Invalid input (empty text, too long, etc.)
- `500 Internal Server Error` - Server-side error

## ⚙️ Configuration

### Adjusting the Threshold

The classification threshold determines when text is labeled as toxic:
- **Lower threshold (0.3)**: More sensitive, catches more potential toxicity
- **Default threshold (0.5)**: Balanced approach
- **Higher threshold (0.7)**: More conservative, fewer false positives

### Redaction Modes

The API supports two redaction modes (configurable in code):
- **token**: Replaces toxic words with `[REDACTED]`
- **mask**: Replaces alphanumeric characters with asterisks

## 🔮 Future Enhancements

- [ ] Better dataset with more diverse examples
- [ ] Transformer-based model (BERT/RoBERTa)
- [ ] Multi-label classification (insult, threat, profanity, etc.)
- [ ] Batch prediction endpoint
- [ ] Rate limiting and authentication
- [ ] Logging and monitoring (Prometheus/Grafana)
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Model versioning and A/B testing

## 📖 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Jigsaw Toxic Comments Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational and portfolio purposes.

## 👤 Author

**Drew Dezco**
- GitHub: [@drewdezco](https://github.com/drewdezco)

---

**Built with ❤️ as a portfolio project demonstrating ML + Software Engineering skills**
