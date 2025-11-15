from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

texts = [
    "You are an idiot!",
    "I hate you so much",
    "Shut up, loser",
    "You're stupid and worthless",
    "Go away, nobody likes you",
    "Thank you for your help",
    "This is a great day",
    "I appreciate your kindness",
    "Have a wonderful time",
    "You did an excellent job"
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(texts, labels)
joblib.dump(pipe, "models/toxicity_model.joblib")
print("✓ Saved toxicity_model.joblib to models/")
