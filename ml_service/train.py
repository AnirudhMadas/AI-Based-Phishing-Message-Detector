import pandas as pd
import numpy as np
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.base import BaseEstimator, TransformerMixin


# LOAD MULTIPLE DATASETS

print("Loading datasets...")

datasets = []

# 1. SMS SPAM DATASET

try:
    sms_df = pd.read_csv(
    "datasets/SMS Spam Collection Dataset.csv",
    encoding="latin-1"
)

    sms_df = sms_df.iloc[:, :2]
    sms_df.columns = ["label", "text"]

    sms_df["label"] = sms_df["label"].map({
        "ham": 0,
        "spam": 1
    })

    datasets.append(sms_df)

    print("SMS Dataset Loaded:", sms_df.shape)

except Exception as e:
    print("SMS Dataset Error:", e)

# 2. PHISHING EMAIL DATASET

try:
    phishing_df = pd.read_csv(
        "datasets/Phishing_Email.csv"
    )

    print("Phishing Email Columns:")
    print(phishing_df.columns)

    # Adjust column names if needed
    text_col = None
    label_col = None

    for col in phishing_df.columns:

        c = col.lower()

        if "text" in c or "email" in c or "body" in c:
            text_col = col

        if "label" in c or "type" in c:
            label_col = col

    phishing_df = phishing_df[[label_col, text_col]]

    phishing_df.columns = ["label", "text"]

    phishing_df["label"] = phishing_df["label"].astype(str).str.lower()

    phishing_df["label"] = phishing_df["label"].apply(
        lambda x: 1 if (
            "phish" in x or
            "spam" in x or
            "fraud" in x or
            "scam" in x
        ) else 0
    )

    datasets.append(phishing_df)

    print("Phishing Email Dataset Loaded:", phishing_df.shape)

except Exception as e:
    print("Phishing Email Dataset Error:", e)

# 3. MALICIOUS URL / PHISH DATASET

try:
    url_df = pd.read_csv(
        "datasets/malicious_phish.csv"
    )

    print("Malicious URL Columns:")
    print(url_df.columns)

    # Common structure:
    # url,type

    url_col = None
    label_col = None

    for col in url_df.columns:

        c = col.lower()

        if "url" in c:
            url_col = col

        if "type" in c or "label" in c:
            label_col = col

    url_df = url_df[[label_col, url_col]]

    url_df.columns = ["label", "text"]

    url_df["label"] = url_df["label"].astype(str).str.lower()

    url_df["label"] = url_df["label"].apply(
        lambda x: 1 if (
            "phish" in x or
            "malicious" in x or
            "defacement" in x
        ) else 0
    )

    datasets.append(url_df)

    print("Malicious URL Dataset Loaded:", url_df.shape)

except Exception as e:
    print("Malicious URL Dataset Error:", e)

# COMBINE DATASETS

df = pd.concat(datasets, ignore_index=True)

df = df.dropna()

df["text"] = df["text"].astype(str)

df = df[df["text"].str.len() > 5]

print("\nCombined Dataset Shape:", df.shape)

# CUSTOM REAL-WORLD PHISHING DATA

phishing_samples = [

    "Your SBI account is blocked verify immediately",
    "URGENT: Your bank KYC expired update now",
    "Click here to claim your cashback reward",
    "Your PayPal account has been suspended",
    "Reset your password immediately",
    "Free recharge available click now",
    "Verify OTP to avoid account suspension",
    "Your package delivery failed update address",
    "A login attempt was detected verify account",
    "You won a free iPhone claim now",
    "Your UPI account will be blocked",
    "Confirm PAN card details immediately",
    "Suspicious login detected click below",
    "Your Aadhaar verification is pending",
    "Limited time banking reward available",

]

safe_samples = [

    "Hey bro where are you",
    "Let's meet tomorrow",
    "Can you send notes",
    "Lunch at 2pm",
    "Happy birthday",
    "Call me later",
    "Reached home safely",
    "Meeting postponed",
    "Good morning",
    "How was your exam",

]

extra_df = pd.DataFrame({

    "label": [1] * len(phishing_samples) + [0] * len(safe_samples),
    "text": phishing_samples + safe_samples

})

df = pd.concat([df, extra_df], ignore_index=True)

# TEXT CLEANING

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Preserve URLs
    text = re.sub(
        r"(https?://\S+|www\.\S+)",
        " URL ",
        text
    )

    # Preserve emails
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

    # Keep important phishing chars
    text = re.sub(
        r"[^a-zA-Z0-9@:/.\s_-]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# FEATURE ENGINEERING

SUSPICIOUS_WORDS = [
    "click",
    "verify",
    "urgent",
    "bank",
    "otp",
    "password",
    "login",
    "claim",
    "winner",
    "limited",
    "suspended",
    "free",
]

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
        1 for word in SUSPICIOUS_WORDS
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

    return pd.Series([
        url_count,
        digit_count,
        upper_count,
        special_count,
        suspicious_count,
        contains_ip,
        contains_shortener
    ])

feature_columns = [
    "url_count",
    "digit_count",
    "upper_count",
    "special_count",
    "suspicious_count",
    "contains_ip",
    "contains_shortener"
]

df[feature_columns] = df["text"].apply(extract_features)


# TRAIN TEST SPLIT

X = df[["clean_text"] + feature_columns]

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)

# PREPROCESSING

preprocessor = ColumnTransformer([

    (
        "text",
        TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True
        ),
        "clean_text"
    ),

    (
        "numeric",
        StandardScaler(),
        feature_columns
    )

])

# MODEL

base_model = LinearSVC(
    class_weight="balanced",
    max_iter=5000
)

model = CalibratedClassifierCV(
    base_model
)

pipeline = Pipeline([

    ("preprocessor", preprocessor),

    ("classifier", model)

])

# TRAIN

print("\nTraining model...")

pipeline.fit(X_train, y_train)

# EVALUATION

print("\nEvaluating model...")

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Safe", "Phishing"]
))

# SAVE MODEL

os.makedirs("model", exist_ok=True)

joblib.dump(
    pipeline,
    "model/phishing_pipeline.pkl"
)

print("\n✅ Model saved successfully!")

# ==========================================
# TEST PREDICTIONS
# ==========================================

test_messages = [

    "Your SBI account has been suspended verify now",

    "Hey are we meeting today?",

    "Claim your free cashback reward immediately",

    "Your OTP is 456789 do not share it",

    "URGENT: Login immediately to avoid suspension"

]

print("SAMPLE PREDICTIONS")

for msg in test_messages:

    cleaned = clean_text(msg)

    features = extract_features(msg)

    sample_df = pd.DataFrame([{

        "clean_text": cleaned,

        "url_count": features[0],
        "digit_count": features[1],
        "upper_count": features[2],
        "special_count": features[3],
        "suspicious_count": features[4],
        "contains_ip": features[5],
        "contains_shortener": features[6]

    }])

    pred = pipeline.predict(sample_df)[0]

    prob = pipeline.predict_proba(sample_df)[0]

    confidence = round(max(prob) * 100, 2)

    label = "PHISHING" if pred == 1 else "SAFE"

    print(f"\nMessage: {msg}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence}%")