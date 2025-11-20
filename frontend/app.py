from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

app = FastAPI(title="Toxicity Classifier Frontend")

# Get the directory of this file
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Get API URL from environment variable or use default
API_URL = os.getenv("API_URL", "http://localhost:8000")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the main HTML page with API URL injected."""
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        # Inject API URL into the HTML
        html_content = html_content.replace(
            "const API_URL = window.API_URL || 'http://localhost:8000';",
            f"const API_URL = window.API_URL || '{API_URL}';"
        )
        return HTMLResponse(content=html_content)

