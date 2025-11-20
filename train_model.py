from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import os

# Get the directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "toxicity_model.joblib")

print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Use comment_text as features and toxic column as labels
texts = df["comment_text"].fillna("").astype(str).tolist()
labels = df["toxic"].astype(int).tolist()

print(f"Loaded {len(texts)} samples")
print(f"Toxic samples: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
print(f"Non-toxic samples: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")

print("\nTraining model...")
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

pipe.fit(texts, labels)
joblib.dump(pipe, MODEL_PATH)
print(f"✓ Saved toxicity_model.joblib to {MODEL_PATH}")
print("✓ Training complete!")
