import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ===============================
# LOAD DATASET
# ===============================

df = pd.read_csv("datasets/spam.csv", encoding="latin-1")
df = df.iloc[:, :2]
df.columns = ["label", "text"]

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna(subset=["text", "label"])

print("Dataset loaded:", df.shape)

# ===============================
# ADD CUSTOM EXAMPLES
# ===============================

additional_phishing = [
    "Your bank account is blocked verify immediately",
    "URGENT: Your account has been suspended. Click here to verify",
    "You've won $5000! Claim your prize now",
    "Your package is waiting. Confirm delivery at this link",
    "Verify your identity or account will be closed today",
]

additional_safe = [
    "hey what's up",
    "ok cool",
    "thanks bro",
    "see you tomorrow",
    "running late, be there soon",
]

additional_df = pd.DataFrame({
    "label": [1] * len(additional_phishing) + [0] * len(additional_safe),
    "text": additional_phishing + additional_safe
})

df = pd.concat([df, additional_df], ignore_index=True)

print("After adding examples:", df.shape)

# ===============================
# CLEAN TEXT
# ===============================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\b\d{10,11}\b", " PHONE ", text)
    text = re.sub(r"\b\d{5}\b", " SHORTCODE ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# ===============================
# TRAIN TEST SPLIT
# ===============================

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# TF-IDF VECTORIZATION
# ===============================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    stop_words="english",
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# MODEL TRAINING
# ===============================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

model.fit(X_train_vec, y_train)

# ===============================
# EVALUATION
# ===============================

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

# ===============================
# SAVE MODEL
# ===============================

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/phishing_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved inside /model folder")
print("🎉 Training complete")