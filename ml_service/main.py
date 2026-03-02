from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

app = FastAPI(title="Phishing Detection API")

# Load model artifacts
model = joblib.load("model/phishing_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


class Message(BaseModel):
    text: str


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\b\d{10,11}\b", " PHONE ", text)
    text = re.sub(r"\b\d{5}\b", " SHORTCODE ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Message):
    clean = clean_text(data.text)
    vec = vectorizer.transform([clean])

    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0].max()

    return {
        "prediction": "phishing" if prediction == 1 else "safe",
        "confidence": round(float(probability), 4)
    }