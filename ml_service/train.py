import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("datasets/spam.csv", encoding="latin-1")

df = df.iloc[:, :2]
df.columns = ["label", "text"]

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna()

print("Dataset loaded:", df.shape)

# =========================
# CUSTOM PHISHING DATA
# =========================

phishing_samples = [
    "Your bank account has been blocked verify immediately",
    "URGENT! Click this link to avoid suspension",
    "Congratulations you won an iPhone click now",
    "Your PayPal account is locked login immediately",
    "Verify your OTP now",
    "Your package is waiting confirm now",
    "Free recharge available click here",
    "You received money claim now",
    "Limited time offer click now",
    "Reset your password immediately",
    "Suspicious login attempt detected verify now",
]

safe_samples = [
    "Hey bro where are you",
    "Let's meet tomorrow",
    "Can you call me later",
    "Thank you",
    "See you soon",
    "Running late",
    "Lunch at 2pm?",
    "Happy birthday",
    "Good morning",
]

extra_df = pd.DataFrame({
    "label": [1] * len(phishing_samples) + [0] * len(safe_samples),
    "text": phishing_samples + safe_samples
})

df = pd.concat([df, extra_df], ignore_index=True)

# =========================
# CLEANING
# =========================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUM ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================
# PIPELINE
# =========================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True
        )
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        )
    )
])

# =========================
# TRAIN
# =========================

pipeline.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Safe", "Phishing"]
))

# =========================
# SAVE
# =========================

os.makedirs("model", exist_ok=True)

joblib.dump(pipeline, "model/phishing_pipeline.pkl")

print("\n✅ Model saved successfully")