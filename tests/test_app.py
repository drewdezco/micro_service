from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app

client = TestClient(app)

def test_health_check():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data

def test_predict_basic():
    resp = client.post("/predict", json={"text": "thank you"})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert "scores" in data
    assert "confidence" in data
    assert data["label"] in ["toxic", "non_toxic"]

def test_predict_toxic():
    resp = client.post("/predict", json={"text": "You are an idiot!"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "toxic"
    assert data["confidence"] > 0.5

def test_predict_with_rationale():
    resp = client.post("/predict?include_rationale=true", json={"text": "You are an idiot!"})
    assert resp.status_code == 200
    data = resp.json()
    assert "rationale" in data
    assert data["rationale"] is not None
    assert len(data["rationale"]) > 0

def test_predict_with_redaction():
    resp = client.post("/predict?include_rationale=true&redact_flagged=true", json={"text": "You are an idiot!"})
    assert resp.status_code == 200
    data = resp.json()
    assert "redacted_text" in data
    assert data["redacted_text"] is not None
    assert "[REDACTED]" in data["redacted_text"]

def test_predict_empty_text():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422
