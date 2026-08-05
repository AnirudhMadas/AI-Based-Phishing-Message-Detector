# 🛡️ AI Phishing Detection System for Mobile Notifications

An AI-powered phishing detection system that analyzes SMS and mobile notifications in real time using a Machine Learning model deployed with FastAPI. The Android application captures incoming notifications, sends them to the ML API, and instantly classifies them as **Safe** or **Phishing**.

---

## 📌 Features

- 🔍 Real-time phishing detection
- 📱 Android notification monitoring
- 🤖 Machine Learning-based classification
- ⚡ FastAPI backend for low-latency predictions
- 📊 Confidence score for every prediction
- 🧹 Text preprocessing before prediction
- 🔒 Detects suspicious URLs and phishing keywords
- 🌐 REST API integration


# 🏗️ Project Architecture

```text
                Incoming SMS / Notification
                           │
                           ▼
                 Android Notification Listener
                           │
                           ▼
                  Text Preprocessing Layer
                           │
                           ▼
                 FastAPI ML Prediction API
                           │
               TF-IDF Feature Extraction
                           │
                           ▼
              Logistic Regression Classifier
                           │
                           ▼
          Safe / Phishing + Confidence Score
                           │
                           ▼
                  Display Result in Android
```

---

# 🚀 Tech Stack

## Frontend

- Android (Java)

## Backend

- FastAPI
- Python

## Machine Learning

- Scikit-Learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib
- Pandas
- NumPy

---

# 📂 Project Structure

```text
AI-Phishing-Detection/
│
├── kotlin-app/
│
├── ml_service/
│   └── datasets/
│   └── model/
│       └── phishing_model.pkl
│       └── tfidf_vectorizer.pkl
│   ├── main.py
│   ├── train.py
│   └── requirements.txt
│
│
│
└── README.md
```

---

# ⚙️ Machine Learning Pipeline

```
Raw Notification
       │
       ▼
Text Cleaning
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Logistic Regression
       │
       ▼
Prediction
       │
       ▼
Confidence Score
```

---

# 📊 Model Information

| Property | Value |
|----------|-------|
| Algorithm | Logistic Regression |
| Vectorizer | TF-IDF |
| Language | Python |
| Framework | Scikit-Learn |
| API | FastAPI |
| Output | Safe / Phishing |
| Serialization | Joblib |

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/AI-Phishing-Detection.git
```

```bash
cd AI-Phishing-Detection
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run FastAPI Server

```bash
uvicorn main:app --reload
```

Server runs at

```
http://127.0.0.1:8000
```

---

# 📡 API Endpoint

## POST `/predict`

### Request

```json
{
  "text": "Your bank account has been suspended. Verify now."
}
```

### Response

```json
{
  "prediction": "Phishing",
  "confidence": 0.96
}
```

---

# 🧠 Dataset

The model is trained using a labeled SMS spam dataset consisting of legitimate and phishing/spam messages.

Example labels:

- Safe (Ham)
- Spam / Phishing

---

# 📈 Workflow

```text
Notification
      │
      ▼
Android App
      │
      ▼
FastAPI API
      │
      ▼
TF-IDF
      │
      ▼
Logistic Regression
      │
      ▼
Prediction
      │
      ▼
Android UI
```

---

# 📋 Future Improvements

- Deep Learning models (LSTM/BERT)
- URL reputation checking
- Sender reputation analysis
- QR code phishing detection
- OCR support for image-based phishing
- Cloud deployment
- Multi-language phishing detection

---

# 👨‍💻 Author

**Anirudh Madas**

B.Tech Information Technology

GitHub: [https://github.com/AnirudhMadas](https://github.com/AnirudhMadas)

LinkedIn: [www.linkedin.com/in/anirudhmadas]( www.linkedin.com/in/anirudhmadas)

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---

# 📄 License

This project is licensed under the MIT License.
