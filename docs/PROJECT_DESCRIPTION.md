# 🧠 Toxicity Classifier API — Beginner-Friendly Walkthrough

Think of this as a small website (an API) you can talk to with code.  
You send it a sentence, and it tells you if the sentence is toxic or not — how sure it is, what words made it think that, and it can even hide (redact) those words.

This project is structured as a mini-build you can complete over **1–2 weeks**.

---

## 🚀 What You’re Building

- A simple program on your computer that listens for requests.
- When you send it text like `"You are an idiot!"`, it replies with:
  - **Label:** `toxic` or `non_toxic`
  - **Confidence score:** how sure it is (e.g., `0.94 = 94%`)
  - **Rationale:** the part of the text that made it think it’s toxic (e.g., `"idiot"`)
  - **Optional redacted version:** `"You are an [REDACTED]!"`

### 💡 Why This Is Cool for Your Portfolio

- Mixes **data science** (training a model) with **software engineering** (serving it as a web API).
- You’ll learn about **APIs**, **machine learning**, and **clean project structure**.

---

## 🗺️ The Plan at a Glance

1. Learn the tools: Python, FastAPI (for the web part), scikit-learn (for the model).  
2. Get a tiny dataset or use a pre-trained simple model.  
3. Build the API with one endpoint: `/predict`.  
4. Add “rationale” (which words mattered) and redaction.  
5. Make it reliable: health check, simple tests, and a README.

---

## 🧰 Step 1: Tools You’ll Use

- **Python:** main language  
- **FastAPI:** for building the web API  
- **scikit-learn:** for training a simple classifier  
- **Joblib:** for saving/loading the model  
- **Requests / cURL / Postman:** for testing the API  

Install them:

```bash
pip install fastapi uvicorn scikit-learn pydantic joblib
```
Absolutely! Here’s your **GitHub-ready Markdown version**, fully formatted with headings, code blocks, and clear structure 👇

---

````markdown
# 🧠 Toxicity Classifier API — Beginner-Friendly Walkthrough

Think of this as a small website (an API) you can talk to with code.  
You send it a sentence, and it tells you if the sentence is toxic or not — how sure it is, what words made it think that, and it can even hide (redact) those words.

This project is structured as a mini-build you can complete over **1–2 weeks**.

---

## 🚀 What You’re Building

- A simple program on your computer that listens for requests.
- When you send it text like `"You are an idiot!"`, it replies with:
  - **Label:** `toxic` or `non_toxic`
  - **Confidence score:** how sure it is (e.g., `0.94 = 94%`)
  - **Rationale:** the part of the text that made it think it’s toxic (e.g., `"idiot"`)
  - **Optional redacted version:** `"You are an [REDACTED]!"`

### 💡 Why This Is Cool for Your Portfolio

- Mixes **data science** (training a model) with **software engineering** (serving it as a web API).
- You’ll learn about **APIs**, **machine learning**, and **clean project structure**.

---

## 🗺️ The Plan at a Glance

1. Learn the tools: Python, FastAPI (for the web part), scikit-learn (for the model).  
2. Get a tiny dataset or use a pre-trained simple model.  
3. Build the API with one endpoint: `/predict`.  
4. Add “rationale” (which words mattered) and redaction.  
5. Make it reliable: health check, simple tests, and a README.

---

## 🧰 Step 1: Tools You’ll Use

- **Python:** main language  
- **FastAPI:** for building the web API  
- **scikit-learn:** for training a simple classifier  
- **Joblib:** for saving/loading the model  
- **Requests / cURL / Postman:** for testing the API  

Install them:

```bash
pip install fastapi uvicorn scikit-learn pydantic joblib
````

Run the server later with:

```bash
uvicorn app:app --reload
```

(That means: “run the Python file `app.py`’s `app` variable as a server.”)

---

## 🧩 Step 2: How the Model Works (Intuitive)

* Turn text into **features** (like word counts).
* Train a **Logistic Regression** classifier on labeled examples (`toxic` vs `non_toxic`).
* The model outputs the probability of toxicity.

Example toy dataset:

| Toxic          | Non-Toxic       |
| -------------- | --------------- |
| shut up        | great job       |
| you are stupid | thank you       |
| idiot          | have a nice day |
| go away loser  | well done       |

You can start with **20–50 examples** and later try the **Jigsaw Toxic Comments** dataset.

---

## 🧠 Step 3: Build a Simple Model

Create `train_model.py`:

```python
# train_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

texts = [
    "you are an idiot",
    "shut up",
    "you are stupid",
    "i hate you",
    "go away loser",
    "great job",
    "thank you",
    "have a nice day",
    "well done",
    "i appreciate your help"
]
labels = [1,1,1,1,1,0,0,0,0,0]  # 1 = toxic, 0 = non_toxic

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), lowercase=True)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(texts, labels)
joblib.dump(pipe, "toxicity_model.joblib")
print("Saved toxicity_model.joblib")
```

Run it:

```bash
python train_model.py
```

---

## 🌐 Step 4: Create the API

Create `app.py`:

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import joblib
import time

app = FastAPI(title="Toxicity Classifier API")

MODEL = joblib.load("toxicity_model.joblib")

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
    toxic_terms = ["idiot", "stupid", "loser", "shut up", "hate"]
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
```

Run it:

```bash
uvicorn app:app --reload
```

Open docs:

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Try it with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict?include_rationale=true&redact_flagged=true" \
  -H "Content-Type: application/json" \
  -d '{"text":"You are an idiot!"}'
```

Expected output:

* `label`: `"toxic"`
* `confidence`: `~1.0`
* `rationale`: `[{"span": "idiot", ...}]`
* `redacted_text`: `"You are an [REDACTED]!"`

---

## 🧪 Step 5: Make It Nicer for Your Portfolio

### README Checklist

* What it does (3–4 sentences)
* How to run it (commands)
* Sample request/response
* Short GIF using `/docs`

### Optional Tests

Create `test_app.py`:

```python
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_basic():
    resp = client.post("/predict", json={"text": "thank you"})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert "scores" in data
```

Run tests:

```bash
pip install pytest
pytest -q
```

Folder ideas:

```
src/
models/
tests/
```

---

## 🔧 Next Steps (When You’re Ready)

* **Better dataset:** Use Jigsaw Toxic Comments subset.
* **Real explanations:** Extract word weights from model or use transformers.
* **Observability:** Track request counts, latency, and add logging.
* **Config:** Move thresholds/redaction mode to a `config.yaml`.

---

## 💬 How to Talk About This in an Interview

* **Problem:** “We needed a quick way to classify text for toxicity and show why.”
* **Approach:** “I trained a small text model and wrapped it in a FastAPI service.”
* **Key Features:** “Returns label, confidence, rationale; optional redaction; health check.”
* **Practices:** “Typed I/O, tests, README.”
* **Next Steps:** “Add better model, metrics, bias checks.”

---

## 🧱 Optional Add-Ons

* 🧩 GitHub repo structure (I can generate this for you)
* 🐳 Docker packaging
* 🤖 Upgrade to a transformer-based model

---

**Enjoy building your Toxicity Classifier API!**

> Small project, big learning payoff 💥




