from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import re
import traceback

# =========================
# APP
# =========================

app = FastAPI(title="AI Phishing Detection API")

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = joblib.load("model/phishing_pipeline.pkl")

print("Model loaded successfully!")

# =========================
# REQUEST MODEL
# =========================

class Message(BaseModel):
    text: str

# =========================
# SUSPICIOUS KEYWORDS
# =========================

SUSPICIOUS_KEYWORDS = [
    "click",
    "verify",
    "urgent",
    "winner",
    "free",
    "claim",
    "password",
    "otp",
    "bank",
    "limited offer",
    "login",
    "suspended",
]

# =========================
# URL DETECTION
# =========================

def contains_url(text):

    url_pattern = r"(https?://\S+|www\.\S+)"

    return bool(re.search(url_pattern, text))

# =========================
# THREAT SCORE
# =========================

def calculate_threat_score(text):

    score = 0

    lower = text.lower()

    for word in SUSPICIOUS_KEYWORDS:

        if word in lower:
            score += 10

    if contains_url(text):
        score += 30

    if len(re.findall(r"\d+", text)) > 3:
        score += 10

    return min(score, 100)

# =========================
# ROUTES
# =========================

@app.get("/")
def root():

    print("Root endpoint called")

    return {
        "status": "running",
        "service": "AI Phishing Detector"
    }

@app.get("/health")
def health():

    print("Health check called")

    return {
        "status": "ok"
    }

# =========================
# PREDICT ROUTE
# =========================

@app.post("/predict")
def predict(data: Message):

    try:

        print("\n========== REQUEST RECEIVED ==========")

        print("Incoming Data:", data)

        text = data.text.strip()

        print("Message Text:", text)

        # =========================
        # MODEL PREDICTION
        # =========================

        prediction = model.predict([text])[0]

        probabilities = model.predict_proba([text])[0]

        confidence = round(float(max(probabilities)) * 100, 2)

        threat_score = calculate_threat_score(text)

        label = "phishing" if prediction == 1 else "safe"

        response = {
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "threat_score": threat_score,
            "contains_url": contains_url(text),
            "message_length": len(text)
        }

        print("Prediction Result:", response)

        print("========== RESPONSE SENT ==========\n")

        return response

    except Exception as e:

        print("\nERROR OCCURRED")
        traceback.print_exc()

        return {
            "success": False,
            "error": str(e)
        }