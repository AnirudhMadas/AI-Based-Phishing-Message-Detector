import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";

dotenv.config();

const app = express();

app.use(express.json());

// =========================
// ENV
// =========================

const PORT = process.env.PORT || 3000;
const ML_API_URL = process.env.ML_API_URL;
const API_KEY = process.env.API_KEY;

if (!ML_API_URL) {
  throw new Error("ML_API_URL missing");
}

if (!API_KEY) {
  throw new Error("API_KEY missing");
}

// =========================
// RATE LIMIT
// =========================

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 300,
});

app.use(limiter);

// =========================
// DUPLICATE FILTER
// =========================

let previousMessage = "";
let previousTime = 0;

function isDuplicate(message) {

  const now = Date.now();

  if (
    message === previousMessage &&
    now - previousTime < 5000
  ) {
    return true;
  }

  previousMessage = message;
  previousTime = now;

  return false;
}

// =========================
// IGNORE PHRASES
// =========================

const SYSTEM_PHRASES = [
  "running in background",
  "checking messages",
  "syncing",
  "accessibility service",
  "notification listener",
];

// =========================
// HEALTH
// =========================

app.get("/health", (req, res) => {

  res.json({
    status: "ok",
    service: "backend"
  });

});

// =========================
// MAIN ROUTE
// =========================

app.post("/notify", async (req, res) => {

  try {

    if (req.headers["x-api-key"] !== API_KEY) {

      return res.status(401).json({
        error: "Unauthorized"
      });

    }

    const {
      body,
      text,
      message,
      title,
      bigText,
      app: appName
    } = req.body;

    const messageText =
      body ||
      text ||
      message ||
      bigText ||
      title ||
      "";

    if (!messageText.trim()) {

      return res.json({
        status: "ignored",
        reason: "empty"
      });

    }

    const lower = messageText.toLowerCase();

    if (
      SYSTEM_PHRASES.some(p => lower.includes(p))
    ) {

      return res.json({
        status: "ignored",
        reason: "system_message"
      });

    }

    if (isDuplicate(messageText)) {

      return res.json({
        status: "ignored",
        reason: "duplicate"
      });

    }

    // =========================
    // CALL ML SERVICE
    // =========================

    const controller = new AbortController();

    const timeout = setTimeout(() => {
      controller.abort();
    }, 8000);

    const response = await fetch(
      `${ML_API_URL}/predict`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          text: messageText
        }),
        signal: controller.signal
      }
    );

    clearTimeout(timeout);

    if (!response.ok) {

      throw new Error(
        `ML Service Error ${response.status}`
      );

    }

    const result = await response.json();

    // =========================
    // FINAL RESPONSE
    // =========================

    return res.json({

      status: "analyzed",

      source_app: appName || "unknown",

      prediction: result.prediction,

      confidence: result.confidence,

      threat_score: result.threat_score,

      contains_url: result.contains_url,

      message: messageText

    });

  } catch (error) {

    console.error("Server Error:", error.message);

    return res.status(500).json({
      status: "error",
      error: error.message
    });

  }

});

// =========================
// START
// =========================

app.listen(PORT, () => {

  console.log(`🚀 Backend running on ${PORT}`);

});