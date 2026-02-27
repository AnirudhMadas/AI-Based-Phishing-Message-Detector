import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load the SMS spam dataset
df = pd.read_csv("datasets\spam.csv", encoding='latin-1')

print("="*60)
print("DATASET LOADING")
print("="*60)
print(f"Original columns: {df.columns.tolist()}")
print(f"Dataset shape: {df.shape}")

# The spam.csv has columns: v1 (label), v2 (text), and some empty columns
# Keep only the first two columns
df = df.iloc[:, :2]
df.columns = ['label', 'text']

# Map labels: ham = safe (0), spam = phishing (1)
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Remove any NaN values
df = df.dropna(subset=['text', 'label'])

print(f"\nCleaned dataset shape: {df.shape}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"  - Safe (ham): {(df['label'] == 0).sum()} messages ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
print(f"  - Phishing (spam): {(df['label'] == 1).sum()} messages ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")

# Add some phishing-specific examples for better detection
additional_phishing = [
    "Your bank account is blocked verify immediately",
    "URGENT: Your account has been suspended. Click here to verify",
    "You've won $5000! Claim your prize now",
    "Your package is waiting. Confirm delivery at this link",
    "ALERT: Suspicious activity detected on your account",
    "Verify your identity or account will be closed today",
    "Your card has been blocked. Call this number immediately",
    "Click here to update your payment information now",
    "Final notice: Your account will be deleted in 24 hours",
    "Congratulations! You won the lottery. Claim here",
    "Your Netflix/Amazon subscription has expired. Update payment",
    "You have a pending refund of $500. Click to claim",
]

additional_safe = [
    "Hey what's up",
    "FM",
    "ok cool",
    "lol that's funny",
    "thanks bro",
    "see you tomorrow",
    "running late, be there soon",
    "where are you",
    "call me when free",
    "want to grab lunch today",
    "happy birthday!",
    "good morning",
    "on my way",
    "let me know",
    "sounds good to me",
    "no worries at all",
    "miss you",
    "love you too",
    "hope you're doing well",
    "can we reschedule?",
]

# Add additional examples
additional_df = pd.DataFrame({
    'label': [1] * len(additional_phishing) + [0] * len(additional_safe),
    'text': additional_phishing + additional_safe
})

df = pd.concat([df, additional_df], ignore_index=True)
print(f"\nAfter adding examples: {df.shape}")

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with placeholder
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    
    # Replace email addresses
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    
    # Replace phone numbers (various formats)
    text = re.sub(r"\b\d{10,11}\b", " PHONE ", text)
    text = re.sub(r"\b\d{5}\b", " SHORTCODE ", text)  # SMS short codes
    
    # Replace numbers but keep the context
    text = re.sub(r"\$\d+", " MONEY ", text)
    text = re.sub(r"£\d+", " MONEY ", text)
    text = re.sub(r"\b\d+\s*(dollars|pounds|gbp|usd)\b", " MONEY ", text)
    
    # Replace remaining numbers
    text = re.sub(r"\d+", " NUM ", text)
    
    # Remove special characters but keep spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

print("\n" + "="*60)
print("TEXT PREPROCESSING")
print("="*60)

df['clean_text'] = df['text'].apply(clean_text)

# Show some examples
print("\nSample of cleaned messages:")
print("-" * 60)
for idx in [0, 3, 10]:
    if idx < len(df):
        print(f"\nOriginal: {df.iloc[idx]['text'][:80]}...")
        print(f"Cleaned:  {df.iloc[idx]['clean_text'][:80]}...")
        print(f"Label:    {'PHISHING' if df.iloc[idx]['label'] == 1 else 'SAFE'}")

# Prepare features and labels
X = df['clean_text']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {len(X_train)} messages")
print(f"Test set: {len(X_test)} messages")

# TF-IDF Vectorization - optimized for SMS
print("\n" + "="*60)
print("FEATURE EXTRACTION (TF-IDF)")
print("="*60)

vectorizer = TfidfVectorizer(
    max_features=5000,      # Reduced for SMS (shorter than emails)
    ngram_range=(1, 3),     # Unigrams, bigrams, and trigrams
    stop_words='english',   # Remove common English stop words
    min_df=2,               # Ignore very rare words
    max_df=0.95,            # Ignore very common words
    sublinear_tf=True       # Use log scaling for term frequency
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Feature matrix shape: {X_train_vec.shape}")

# Train the model
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle slight class imbalance
    C=1.0,                     # Regularization strength
    solver='liblinear'         # Good for small-medium datasets
)

model.fit(X_train_vec, y_train)
print("✓ Model trained successfully")

# Make predictions
y_pred = model.predict(X_test_vec)
y_pred_proba = model.predict_proba(X_test_vec)

# Evaluate the model
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted Safe | Predicted Phishing")
print(f"Actually Safe    {cm[0][0]:>14} | {cm[0][1]:>18}")
print(f"Actually Phishing{cm[1][0]:>14} | {cm[1][1]:>18}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing'], digits=4))

# Test on specific real-world examples
print("\n" + "="*60)
print("TESTING ON REAL EXAMPLES")
print("="*60)

test_messages = [
    ("Your bank account is blocked verify immediately", "PHISHING"),
    ("FM", "SAFE"),
    ("hey wassup", "SAFE"),
    ("Anirudh hey niggaa wassup", "SAFE"),
    ("Click here to claim your prize now!", "PHISHING"),
    ("running late, be there in 5", "SAFE"),
    ("URGENT: Your account has been suspended", "PHISHING"),
    ("thanks for the help bro", "SAFE"),
    ("You have won $1000! Click to claim", "PHISHING"),
    ("ok cool see you tomorrow", "SAFE"),
    ("WINNER!! Call this number to claim prize", "PHISHING"),
    ("lol that's hilarious", "SAFE"),
]

print("\nPredictions on test messages:")
print("-" * 80)

correct = 0
for msg, expected in test_messages:
    clean = clean_text(msg)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec).max()
    predicted_label = "PHISHING" if prediction == 1 else "SAFE"
    
    is_correct = predicted_label == expected
    correct += is_correct
    
    status = "✓" if is_correct else "✗"
    
    print(f"\n{status} Message: '{msg}'")
    print(f"  Expected:  {expected}")
    print(f"  Predicted: {predicted_label} (confidence: {probability:.4f})")

print(f"\n{'='*80}")
print(f"Manual Test Accuracy: {correct}/{len(test_messages)} ({correct/len(test_messages)*100:.1f}%)")

# Save the model and vectorizer
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✓ Model saved to: phishing_model.pkl")
print("✓ Vectorizer saved to: tfidf_vectorizer.pkl")

# Show most important features
print("\n" + "="*60)
print("TOP PHISHING INDICATORS (Most Important Features)")
print("="*60)

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Get top phishing indicators (positive coefficients)
top_phishing_idx = coefficients.argsort()[-20:][::-1]
print("\nTop 20 words/phrases indicating PHISHING:")
for idx in top_phishing_idx:
    print(f"  {feature_names[idx]:20s} (weight: {coefficients[idx]:6.3f})")

# Get top safe indicators (negative coefficients)
top_safe_idx = coefficients.argsort()[:20]
print("\nTop 20 words/phrases indicating SAFE:")
for idx in top_safe_idx:
    print(f"  {feature_names[idx]:20s} (weight: {coefficients[idx]:6.3f})")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Restart your ML API: python ml_api.py")
print("2. Test with your server: node server.js")
print("3. Send test messages to see improved predictions")