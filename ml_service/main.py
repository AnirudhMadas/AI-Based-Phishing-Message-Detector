from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd
import re
import traceback

# ==========================================
# APP
# ==========================================

app = FastAPI(title="AI Phishing Detection API")

# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = joblib.load("model/phishing_pipeline.pkl")

print("Model loaded successfully!")

# ==========================================
# REQUEST MODEL
# ==========================================

class Message(BaseModel):
    text: str

# ==========================================
# SUSPICIOUS KEYWORDS
# ==========================================

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

# ==========================================
# CLEAN TEXT
# ==========================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Replace URLs
    text = re.sub(
        r"(https?://\S+|www\.\S+)",
        " URL ",
        text
    )

    # Replace emails
    text = re.sub(
        r"\S+@\S+",
        " EMAIL ",
        text
    )

    # Replace numbers
    text = re.sub(
        r"\d+",
        " NUM ",
        text
    )

    # Preserve phishing-related symbols
    text = re.sub(
        r"[^a-zA-Z0-9@:/.\s_-]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ==========================================
# URL DETECTION
# ==========================================

def contains_url(text):

    url_pattern = r"(https?://\S+|www\.\S+)"

    return bool(re.search(url_pattern, text))

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(text):

    text = str(text)

    lower = text.lower()

    url_count = len(
        re.findall(
            r"(https?://\S+|www\.\S+)",
            text
        )
    )

    digit_count = len(
        re.findall(r"\d", text)
    )

    upper_count = sum(
        1 for c in text if c.isupper()
    )

    special_count = len(
        re.findall(r"[!@#$%^&*()_+=]", text)
    )

    suspicious_count = sum(
        1 for word in SUSPICIOUS_KEYWORDS
        if word in lower
    )

    contains_ip = int(bool(
        re.search(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            text
        )
    ))

    contains_shortener = int(bool(
        re.search(
            r"(bit\.ly|tinyurl|goo\.gl|t\.co)",
            lower
        )
    ))

    return {
        "url_count": url_count,
        "digit_count": digit_count,
        "upper_count": upper_count,
        "special_count": special_count,
        "suspicious_count": suspicious_count,
        "contains_ip": contains_ip,
        "contains_shortener": contains_shortener
    }

# ==========================================
# THREAT SCORE
# ==========================================

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

    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
        score += 20

    if re.search(r"(bit\.ly|tinyurl|goo\.gl|t\.co)", lower):
        score += 20

    return min(score, 100)

# ==========================================
# ROUTES
# ==========================================

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

# ==========================================
# PREDICT ROUTE
# ==========================================

@app.post("/predict")
def predict(data: Message):

    try:

        print("\n========== REQUEST RECEIVED ==========")

        print("Incoming Data:", data)

        text = data.text.strip()

        print("Message Text:", text)

        # ==========================================
        # CLEAN + FEATURE EXTRACTION
        # ==========================================

        cleaned = clean_text(text)

        features = extract_features(text)

        # ==========================================
        # CREATE INPUT DATAFRAME
        # ==========================================

        input_df = pd.DataFrame([{

            "clean_text": cleaned,

            "url_count": features["url_count"],
            "digit_count": features["digit_count"],
            "upper_count": features["upper_count"],
            "special_count": features["special_count"],
            "suspicious_count": features["suspicious_count"],
            "contains_ip": features["contains_ip"],
            "contains_shortener": features["contains_shortener"]

        }])

        # ==========================================
        # MODEL PREDICTION
        # ==========================================

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        confidence = round(float(max(probabilities)) * 100, 2)

        threat_score = calculate_threat_score(text)

        label = "phishing" if prediction == 1 else "safe"

        # ==========================================
        # EXPLAINABILITY
        # ==========================================

        reasons = []

        if contains_url(text):
            reasons.append("Contains suspicious URL")

        if "otp" in text.lower():
            reasons.append("Contains OTP-related content")

        if "bank" in text.lower():
            reasons.append("Contains banking keywords")

        if "verify" in text.lower():
            reasons.append("Requests verification")

        if "urgent" in text.lower():
            reasons.append("Uses urgent language")

        if features["contains_shortener"]:
            reasons.append("Contains shortened URL")

        if features["contains_ip"]:
            reasons.append("Contains IP-based URL")

        # ==========================================
        # RESPONSE
        # ==========================================

        response = {
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "threat_score": threat_score,
            "contains_url": contains_url(text),
            "message_length": len(text),
            "reasons": reasons
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