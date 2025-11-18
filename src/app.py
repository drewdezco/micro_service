from fastapi import FastAPI, Query
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
