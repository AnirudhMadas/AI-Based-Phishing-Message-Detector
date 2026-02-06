from fastapi import FastAPI
import joblib
import re

app = FastAPI(title="Phishing Detection AI")

# Load trained artifacts
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()


@app.post("/predict")
def predict(data: dict):
    text = data.get("text", "")

    if not text:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "reason": "Empty input"
        }

    clean = clean_text(text)
    vec = vectorizer.transform([clean])

    prediction = int(model.predict(vec)[0])
    probability = float(model.predict_proba(vec).max())

    return {
        "label": "phishing" if prediction == 1 else "safe",
        "confidence": round(probability, 4)
    }
