from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import time
import os

app = FastAPI(title="Toxicity Classifier API")

# Add CORS middleware to allow frontend to make requests
# Must be added before routes are defined
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def find_rationale_spans(text: str, top_k: int = 5):
    """
    Find toxic terms in text by using the model's feature importance.
    Returns spans of words that contribute most to toxic classification.
    """
    import re
    
    try:
        # Get the vectorizer and classifier from the pipeline
        vectorizer = MODEL.named_steps['tfidf']
        classifier = MODEL.named_steps['clf']
        
        # Transform the text to get feature names
        X = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients for toxic class (class 1)
        if hasattr(classifier, 'coef_'):
            coef = classifier.coef_[0]  # Get coefficients for toxic class
        else:
            return _fallback_rationale(text, top_k)
        
        # Common stop words and non-toxic words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
            'our', 'their', 'think', 'think', 'great', 'guy', 'sometimes', 'absolute'
        }
        
        # Get feature indices and their importance scores
        feature_scores = []
        for i in range(X.shape[1]):
            if X[0, i] > 0:  # Only consider features present in the text
                score = coef[i] * X[0, i]
                # Only include features with positive contribution to toxicity
                # and filter out common stop words
                term = feature_names[i]
                if score > 0.1 and term.lower() not in stop_words:  # Minimum threshold
                    feature_scores.append((i, score, term))
        
        # Sort by importance (highest scores first)
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top features
        top_features = feature_scores[:top_k]
        
        # Tokenize text to find word boundaries
        words = re.findall(r'\b\w+\b', text)
        word_positions = []
        current_pos = 0
        for word in words:
            pos = text.find(word, current_pos)
            if pos >= 0:
                word_positions.append((word.lower(), pos, pos + len(word)))
                current_pos = pos + len(word)
        
        # Find these terms in the original text using word boundaries
        spans = []
        lower_text = text.lower()
        
        for feat_idx, score, term in top_features:
            # Try to find as a complete word (not substring)
            term_lower = term.lower()
            
            # Check if it's a single word or n-gram
            if ' ' in term:
                # For n-grams, find the phrase
                i = lower_text.find(term_lower)
                if i >= 0:
                    actual_span = text[i:i+len(term)]
                    spans.append(RationaleSpan(
                        span=actual_span,
                        start=i,
                        end=i+len(term),
                        weight=float(score)
                    ))
            else:
                # For single words, find using word boundaries
                for word_lower, start, end in word_positions:
                    if word_lower == term_lower:
                        actual_span = text[start:end]
                        spans.append(RationaleSpan(
                            span=actual_span,
                            start=start,
                            end=end,
                            weight=float(score)
                        ))
                        break  # Only add once
        
        # Remove duplicates and sort by position
        seen = set()
        unique_spans = []
        for span in spans:
            key = (span.start, span.end)
            if key not in seen:
                seen.add(key)
                unique_spans.append(span)
        
        return unique_spans[:top_k] if unique_spans else None
        
    except Exception as e:
        # Fallback to simple keyword matching if model introspection fails
        return _fallback_rationale(text, top_k)

def _fallback_rationale(text: str, top_k: int = 5):
    """Fallback method using common toxic terms with word boundary matching."""
    import re
    
    toxic_terms = [
        "idiot", "stupid", "loser", "hate", "worthless", "dick", "fuck", 
        "asshole", "bastard", "bitch", "damn", "hell", "crap", "shit",
        "dickhead", "moron", "retard", "fool", "jerk", "scum", "ass",
        "fucking", "damned", "hated", "stupid", "idiotic"
    ]
    
    lower = text.lower()
    spans = []
    
    # Use word boundaries to match whole words only
    for term in toxic_terms:
        # Create a regex pattern with word boundaries
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        matches = list(re.finditer(pattern, lower))
        
        for match in matches:
            start = match.start()
            end = match.end()
            actual_span = text[start:end]
            spans.append(RationaleSpan(
                span=actual_span,
                start=start,
                end=end,
                weight=1.0
            ))
            if len(spans) >= top_k:
                break
        
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
