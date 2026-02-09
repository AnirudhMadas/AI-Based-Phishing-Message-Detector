from fastapi import FastAPI
import joblib
import re

app = FastAPI(title="Phishing Detection AI")

# Load trained artifacts
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Confidence thresholds
PHISHING_THRESHOLD = 0.65  # Need 65% confidence to flag as phishing
SAFE_THRESHOLD = 0.55       # Need 55% confidence to flag as safe


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

    # Handle very short messages (1-2 characters)
    if len(text.strip()) <= 2:
        return {
            "label": "safe",
            "confidence": 0.95,
            "reason": "Too short to be phishing"
        }

    clean = clean_text(text)
    vec = vectorizer.transform([clean])

    # Get probabilities for both classes
    probabilities = model.predict_proba(vec)[0]
    prob_safe = probabilities[0]
    prob_phishing = probabilities[1]
    
    # Determine prediction with threshold
    if prob_phishing >= PHISHING_THRESHOLD:
        label = "phishing"
        confidence = prob_phishing
    elif prob_safe >= SAFE_THRESHOLD:
        label = "safe"
        confidence = prob_safe
    else:
        # Low confidence - mark as uncertain
        label = "uncertain"
        confidence = max(prob_safe, prob_phishing)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "safe": round(prob_safe, 4),
            "phishing": round(prob_phishing, 4)
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)