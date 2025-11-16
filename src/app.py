from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import time
import os

app = FastAPI(title="Toxicity Classifier API")

# get the directory of this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# get the path to the model
MODEL_PATH = os.path.join(BASE_DIR, "models", "toxicity_model.joblib")

# load the model
MODEL = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class RationaleSpan(BaseModel):
    span: str
    start: int
    end: int
    weight: float

class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict
    rationale: list[RationaleSpan] | None = None
    redacted_text: str | None = None
    meta: dict

def simple_scores(text: str):
    proba = MODEL.predict_proba([text])[0]
    return {"non_toxic": float(proba[0]), "toxic": float(proba[1])}

def choose_label(scores: dict, threshold: float):
    return "toxic" if scores["toxic"] >= threshold else "non_toxic"

def find_rationale_spans(text: str, top_k: int = 1):
    toxic_terms = ["idiot", "stupid", "loser", "shut up", "hate", "worthless"]
    lower = text.lower()
    spans = []
    for term in toxic_terms:
        i = lower.find(term)
        if i >= 0:
            spans.append(RationaleSpan(span=text[i:i+len(term)], start=i, end=i+len(term), weight=1.0))
            if len(spans) >= top_k:
                break
    return spans if spans else None

def redact(text: str, spans: list[RationaleSpan] | None, mode: str = "token") -> str | None:
    if not spans:
        return None
    chars = list(text)
    for s in sorted(spans, key=lambda x: x.start, reverse=True):
        segment = text[s.start:s.end]
        replacement = "[REDACTED]" if mode == "token" else "".join("*" if c.isalnum() else c for c in segment)
        chars[s.start:s.end] = list(replacement)
    return "".join(chars)

@app.get("/", response_class=HTMLResponse)
def ui():
    """Serve the user interface for the toxicity classifier."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Toxicity Classifier</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: #1a1a1a;
                color: #d4d4d4;
                min-height: 100vh;
                padding: 20px;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .container {
                background: #2b2b2b;
                border-radius: 8px;
                border: 1px solid #3b3b3b;
                max-width: 800px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                color: #ffffff;
                margin-bottom: 10px;
                font-size: 2em;
                font-weight: 600;
            }
            .subtitle {
                color: #9ca3af;
                margin-bottom: 30px;
                font-size: 0.95em;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #d4d4d4;
                font-weight: 500;
                font-size: 0.95em;
            }
            textarea {
                width: 100%;
                padding: 12px;
                background: #1a1a1a;
                border: 1px solid #3b3b3b;
                border-radius: 4px;
                font-size: 1em;
                font-family: inherit;
                color: #d4d4d4;
                resize: vertical;
                min-height: 120px;
                transition: border-color 0.3s;
            }
            textarea:focus {
                outline: none;
                border-color: #00d4aa;
            }
            textarea::placeholder {
                color: #6b7280;
            }
            .options {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .option-group {
                display: flex;
                flex-direction: column;
            }
            input[type="number"] {
                padding: 10px;
                background: #1a1a1a;
                border: 1px solid #3b3b3b;
                border-radius: 4px;
                font-size: 0.95em;
                color: #d4d4d4;
                transition: border-color 0.3s;
            }
            input[type="number"]:focus {
                outline: none;
                border-color: #00d4aa;
            }
            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            input[type="checkbox"] {
                width: 18px;
                height: 18px;
                cursor: pointer;
                accent-color: #00d4aa;
            }
            button {
                background: #00d4aa;
                color: #1a1a1a;
                border: none;
                padding: 14px 28px;
                border-radius: 4px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: background-color 0.2s, transform 0.2s;
            }
            button:hover {
                background: #00b894;
                transform: translateY(-1px);
            }
            button:active {
                transform: translateY(0);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            #results {
                margin-top: 30px;
                padding: 20px;
                background: #1a1a1a;
                border-radius: 4px;
                border: 1px solid #3b3b3b;
                display: none;
            }
            .result-label {
                font-size: 1.3em;
                font-weight: 600;
                margin-bottom: 15px;
                padding: 12px;
                border-radius: 4px;
                text-align: center;
            }
            .toxic {
                background: rgba(239, 68, 68, 0.2);
                color: #f87171;
                border: 1px solid rgba(239, 68, 68, 0.3);
            }
            .non-toxic {
                background: rgba(34, 197, 94, 0.2);
                color: #4ade80;
                border: 1px solid rgba(34, 197, 94, 0.3);
            }
            .result-section {
                margin-bottom: 20px;
            }
            .result-section h3 {
                color: #ffffff;
                margin-bottom: 10px;
                font-size: 1.1em;
                font-weight: 600;
            }
            .scores {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }
            .score-item {
                flex: 1;
                min-width: 150px;
                padding: 12px;
                background: #2b2b2b;
                border: 1px solid #3b3b3b;
                border-radius: 4px;
                text-align: center;
            }
            .score-label {
                font-size: 0.85em;
                color: #9ca3af;
                margin-bottom: 5px;
                text-transform: uppercase;
            }
            .score-value {
                font-size: 1.3em;
                font-weight: 600;
                color: #ffffff;
            }
            .rationale-list {
                list-style: none;
                padding: 0;
            }
            .rationale-item {
                padding: 10px;
                background: rgba(251, 191, 36, 0.15);
                border-left: 4px solid #fbbf24;
                margin-bottom: 8px;
                border-radius: 4px;
                color: #fbbf24;
            }
            .redacted-text {
                padding: 12px;
                background: #1a1a1a;
                border: 1px solid #3b3b3b;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                border-left: 4px solid #6b7280;
                color: #9ca3af;
            }
            .meta-info {
                font-size: 0.9em;
                color: #9ca3af;
                padding-top: 15px;
                border-top: 1px solid #3b3b3b;
            }
            .error {
                background: rgba(239, 68, 68, 0.2);
                color: #f87171;
                padding: 15px;
                border-radius: 4px;
                border: 1px solid rgba(239, 68, 68, 0.3);
                margin-top: 20px;
                display: none;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #9ca3af;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Toxicity Classifier</h1>
            <p class="subtitle">Enter text below to check if it's toxic or non-toxic</p>
            
            <form id="classifyForm">
                <div class="form-group">
                    <label for="text">Text to Classify:</label>
                    <textarea id="text" name="text" placeholder="Enter text here..." required></textarea>
                </div>
                
                <div class="options">
                    <div class="option-group">
                        <label for="threshold">Threshold (0.0 - 1.0):</label>
                        <input type="number" id="threshold" name="threshold" value="0.5" min="0" max="1" step="0.1">
                    </div>
                    <div class="option-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="include_rationale" name="include_rationale" checked>
                            <label for="include_rationale">Include Rationale</label>
                        </div>
                    </div>
                    <div class="option-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="redact_flagged" name="redact_flagged">
                            <label for="redact_flagged">Redact Flagged Text</label>
                        </div>
                    </div>
                </div>
                
                <button type="submit" id="submitBtn">Classify Text</button>
            </form>
            
            <div class="loading" id="loading">Analyzing text...</div>
            <div class="error" id="error"></div>
            
            <div id="results">
                <div class="result-label" id="resultLabel"></div>
                
                <div class="result-section">
                    <h3>Confidence Scores</h3>
                    <div class="scores" id="scores"></div>
                </div>
                
                <div class="result-section" id="rationaleSection" style="display: none;">
                    <h3>Rationale (Flagged Terms)</h3>
                    <ul class="rationale-list" id="rationaleList"></ul>
                </div>
                
                <div class="result-section" id="redactedSection" style="display: none;">
                    <h3>Redacted Text</h3>
                    <div class="redacted-text" id="redactedText"></div>
                </div>
                
                <div class="meta-info" id="metaInfo"></div>
            </div>
        </div>
        
        <script>
            const form = document.getElementById('classifyForm');
            const results = document.getElementById('results');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const submitBtn = document.getElementById('submitBtn');
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                // Hide previous results and errors
                results.style.display = 'none';
                error.style.display = 'none';
                loading.style.display = 'block';
                submitBtn.disabled = true;
                
                const text = document.getElementById('text').value;
                const threshold = parseFloat(document.getElementById('threshold').value);
                const include_rationale = document.getElementById('include_rationale').checked;
                const redact_flagged = document.getElementById('redact_flagged').checked;
                
                // Build query parameters
                const params = new URLSearchParams({
                    include_rationale: include_rationale.toString(),
                    redact_flagged: redact_flagged.toString(),
                    threshold: threshold.toString()
                });
                
                try {
                    const response = await fetch(`/predict?${params.toString()}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: text })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    // Display results
                    displayResults(data);
                    
                } catch (err) {
                    error.textContent = `Error: ${err.message}`;
                    error.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                    submitBtn.disabled = false;
                }
            });
            
            function displayResults(data) {
                // Set label
                const resultLabel = document.getElementById('resultLabel');
                resultLabel.textContent = `Label: ${data.label.toUpperCase()}`;
                resultLabel.className = `result-label ${data.label === 'toxic' ? 'toxic' : 'non-toxic'}`;
                
                // Display scores
                const scoresDiv = document.getElementById('scores');
                scoresDiv.innerHTML = '';
                for (const [key, value] of Object.entries(data.scores)) {
                    const scoreItem = document.createElement('div');
                    scoreItem.className = 'score-item';
                    scoreItem.innerHTML = `
                        <div class="score-label">${key.replace('_', ' ').toUpperCase()}</div>
                        <div class="score-value">${(value * 100).toFixed(1)}%</div>
                    `;
                    scoresDiv.appendChild(scoreItem);
                }
                
                // Display rationale
                const rationaleSection = document.getElementById('rationaleSection');
                const rationaleList = document.getElementById('rationaleList');
                if (data.rationale && data.rationale.length > 0) {
                    rationaleSection.style.display = 'block';
                    rationaleList.innerHTML = '';
                    data.rationale.forEach(span => {
                        const li = document.createElement('li');
                        li.className = 'rationale-item';
                        li.textContent = `"${span.span}" (position ${span.start}-${span.end})`;
                        rationaleList.appendChild(li);
                    });
                } else {
                    rationaleSection.style.display = 'none';
                }
                
                // Display redacted text
                const redactedSection = document.getElementById('redactedSection');
                const redactedText = document.getElementById('redactedText');
                if (data.redacted_text) {
                    redactedSection.style.display = 'block';
                    redactedText.textContent = data.redacted_text;
                } else {
                    redactedSection.style.display = 'none';
                }
                
                // Display meta info
                const metaInfo = document.getElementById('metaInfo');
                metaInfo.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}% | Latency: ${data.meta.latency_ms}ms | Threshold: ${data.meta.threshold_used}`;
                
                results.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/healthz")
def health():
    return {"status": "ok", "model_version": "toy-0.1"}

@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    include_rationale: bool = Query(True),
    redact_flagged: bool = Query(False),
    threshold: float = Query(0.5, ge=0, le=1),
):
    t0 = time.time()
    scores = simple_scores(req.text)
    label = choose_label(scores, threshold)
    rationale = find_rationale_spans(req.text) if include_rationale else None
    redacted = redact(req.text, rationale) if (redact_flagged and rationale) else None
    latency_ms = (time.time() - t0) * 1000.0

    return PredictResponse(
        label=label,
        confidence=scores[label],
        scores=scores,
        rationale=rationale,
        redacted_text=redacted,
        meta={"latency_ms": round(latency_ms, 2), "threshold_used": threshold},
    )
